"""Song analysis: settings, model backends, the analysis result and the pipeline."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio_io, model_store, models, rhythm, segments, theory
from .models import ModelSpec
from .theory import Chord, Key

SAMPLE_RATE = 22050      # every model here works at 22.05 kHz
RESULT_VERSION = 1


class Cancelled(Exception):
    pass


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Chordfinder"


def default_models_dir() -> Path:
    return app_data_dir() / "models"


def default_cache_dir() -> Path:
    return app_data_dir() / "results"


@dataclass
class Settings:
    chord_model: str = models.DEFAULTS[models.CHORDS]
    beat_model: str = models.DEFAULTS[models.BEATS]
    key_model: str = models.DEFAULTS[models.KEY]

    snap_to_beats: bool = True        # move chord changes onto the nearest beat
    min_chord_seconds: float = 0.4    # shorter chords are merged into a neighbour
    tuning_correction: bool = True    # retune the analysis to the song's reference pitch

    use_cache: bool = True            # reuse earlier results for unchanged files
    save_chart: bool = False          # write <song>.chords.txt next to the song

    device: str = "cuda"
    models_dir: Path = field(default_factory=default_models_dir)
    cache_dir: Path = field(default_factory=default_cache_dir)
    allow_download: bool = True

    def spec(self, task: str) -> ModelSpec:
        model_id = {models.CHORDS: self.chord_model, models.BEATS: self.beat_model,
                    models.KEY: self.key_model}[task]
        return models.get(model_id, task)

    def specs(self) -> list[ModelSpec]:
        return [self.spec(t) for t in (models.CHORDS, models.BEATS, models.KEY)]


@dataclass
class Callbacks:
    log: Callable[[str], None] = print
    progress: Callable[[float, str], None] = lambda fraction, stage: None
    is_cancelled: Callable[[], bool] = lambda: False


# ==================================================================== result
@dataclass
class Analysis:
    path: str
    duration: float
    bpm: float
    bpm_low: float
    bpm_high: float
    bpm_stable: bool
    meter: int | None
    beats: list[float]
    downbeats: list[float]
    tuning_cents: float
    key_index: int                      # Key.index of the detected key
    key_probs: list[float]              # 24 values, indexed like Key.index
    chords: list[tuple[float, float, str]]
    models: dict[str, str]
    options: dict
    elapsed: float = 0.0
    version: int = RESULT_VERSION

    # ------------------------------------------------------------ helpers
    @property
    def key(self) -> Key:
        return Key.from_index(self.key_index)

    @property
    def tempo(self) -> rhythm.Tempo:
        return rhythm.Tempo(self.bpm, self.bpm_low, self.bpm_high, self.bpm_stable)

    @property
    def a4_hz(self) -> float:
        return rhythm.cents_to_a4(self.tuning_cents)

    @property
    def key_confidence(self) -> float:
        return float(self.key_probs[self.key_index]) if self.key_probs else 0.0

    def key_candidates(self, n: int = 3) -> list[tuple[Key, float]]:
        order = np.argsort(self.key_probs)[::-1][:n]
        return [(Key.from_index(int(i)), float(self.key_probs[i])) for i in order]

    @property
    def meter_text(self) -> str:
        return f"{self.meter}/4" if self.meter else "–"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(text: str) -> "Analysis":
        d = json.loads(text)
        if d.get("version") != RESULT_VERSION:
            raise ValueError("Result from an incompatible version.")
        d["chords"] = [tuple(c) for c in d["chords"]]
        return Analysis(**d)

    # ------------------------------------------------------------ exports
    def chart(self, key: Key | None = None, with_functions: bool = True) -> str:
        """Human-readable chord chart."""
        key = key or self.key
        lines = [
            f"{Path(self.path).name}",
            f"Key: {self.key.name} ({self.key.camelot})   Tempo: {self.tempo.text()}   Meter: {self.meter_text}"
            f"   Tuning: {rhythm.describe_tuning(self.tuning_cents)}",
            "",
        ]
        if key != self.key:
            lines.insert(2, f"Functions relative to: {key.name}")
        for start, end, label in self.chords:
            chord = Chord.parse(label)
            if not chord.is_chord:
                continue
            m, s = divmod(start, 60)
            lines.append(f"{int(m):2d}:{s:05.2f}  {theory.describe(chord, key, with_functions)}")
        return "\n".join(lines) + "\n"

    def lab(self) -> str:
        """MIREX .lab format: start end label (one chord per line)."""
        return "".join(f"{s:.3f}\t{e:.3f}\t{lab}\n" for s, e, lab in self.chords)


# ===================================================================== cache
def cache_key(path: Path, s: Settings) -> str:
    st = path.stat()
    ident = json.dumps([str(path.resolve()).lower(), st.st_size, int(st.st_mtime), s.chord_model, s.beat_model,
                        s.key_model, s.snap_to_beats, round(s.min_chord_seconds, 3), s.tuning_correction,
                        RESULT_VERSION])
    return hashlib.sha1(ident.encode()).hexdigest()


def load_cached(path: Path, s: Settings) -> Analysis | None:
    try:
        f = Path(s.cache_dir) / f"{cache_key(path, s)}.json"
        return Analysis.from_json(f.read_text(encoding="utf-8")) if f.is_file() else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def store_cached(path: Path, s: Settings, result: Analysis) -> None:
    try:
        Path(s.cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(s.cache_dir) / f"{cache_key(path, s)}.json").write_text(result.to_json(), encoding="utf-8")
    except OSError:
        pass  # the cache is only a convenience


# ================================================================== backends
def _device(name: str) -> str:
    import torch
    return "cuda" if name == "cuda" and torch.cuda.is_available() else "cpu"


class ChordBackend:
    """BTC chord recognition on half-overlapping 10 s windows."""

    def __init__(self, spec: ModelSpec, models_dir: Path, device: str):
        from .nets import btc
        self.btc = btc
        self.device = _device(device)
        self.num_chords = int(spec.param("num_chords"))
        self.model, self.mean, self.std = btc.load_checkpoint(model_store.file_path(models_dir, spec),
                                                              self.num_chords, self.device)
        self.labels = btc.labels(self.num_chords)

    def features(self, y: np.ndarray, tuning_cents: float) -> np.ndarray:
        """Log-magnitude CQT, computed in 10 s pieces exactly like the training data. (frames, 144)"""
        import librosa
        b = self.btc
        tuning = tuning_cents / (1200 / b.BINS_PER_OCTAVE)   # in fractions of a CQT bin
        chunk = int(b.SAMPLE_RATE * b.CHUNK_SECONDS)
        parts = []
        for start in range(0, max(len(y), 1), chunk):
            piece = y[start:start + chunk]
            if len(piece) < b.HOP_LENGTH:  # librosa needs a little audio; pad the very end
                piece = np.pad(piece, (0, b.HOP_LENGTH - len(piece)))
            parts.append(librosa.cqt(piece, sr=b.SAMPLE_RATE, n_bins=b.N_BINS, bins_per_octave=b.BINS_PER_OCTAVE,
                                     hop_length=b.HOP_LENGTH, tuning=tuning))
        return np.abs(np.concatenate(parts, axis=1)).T

    def predict(self, magnitude: np.ndarray, is_cancelled: Callable[[], bool],
                progress: Callable[[float], None]) -> list[str]:
        import torch
        t = self.btc.TIMESTEP
        feats = (np.log(magnitude + 1e-6) - self.mean) / self.std
        n = len(feats)
        hop = t // 2
        # Every frame is seen by two half-overlapping windows; padding (zeros = the feature mean)
        # makes the windows tile the song exactly.
        padded = np.pad(feats, ((hop, hop + (-n) % hop), (0, 0)))
        probs = np.zeros((len(padded), self.num_chords), np.float32)
        x = torch.tensor(padded, dtype=torch.float32, device=self.device)
        starts = list(range(0, len(padded) - t + 1, hop))
        weight = np.hanning(t + 2)[1:-1].astype(np.float32)[:, None]
        batch = 32
        with torch.inference_mode():
            for i in range(0, len(starts), batch):
                if is_cancelled():
                    raise Cancelled()
                idx = starts[i:i + batch]
                window = torch.stack([x[s:s + t] for s in idx])
                p = torch.softmax(self.model(window), dim=-1).cpu().numpy()
                # Frames near a window edge get less weight: the model sees less context there.
                for s, pw in zip(idx, p):
                    probs[s:s + t] += pw * weight
                progress(min(1.0, (i + batch) / len(starts)))
        probs = probs[hop:hop + n]
        return [self.labels[i] for i in probs.argmax(axis=1)]

    def close(self) -> None:
        del self.model


class BeatThisBackend:
    def __init__(self, spec: ModelSpec, models_dir: Path, device: str):
        from beat_this.inference import Audio2Beats
        dev = _device(device)
        self.tracker = Audio2Beats(checkpoint_path=str(model_store.file_path(models_dir, spec)), device=dev,
                                   float16=dev == "cuda")

    def track(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beats, downbeats = self.tracker(y.astype(np.float64), SAMPLE_RATE)
        return np.asarray(beats, float), np.asarray(downbeats, float)

    def close(self) -> None:
        del self.tracker


class LibrosaBeatBackend:
    def __init__(self, spec: ModelSpec, models_dir: Path, device: str):
        pass

    def track(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import librosa
        _, frames = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE, hop_length=512, units="frames")
        return librosa.frames_to_time(frames, sr=SAMPLE_RATE, hop_length=512), np.zeros(0)

    def close(self) -> None:
        pass


class SKeyBackend:
    def __init__(self, spec: ModelSpec, models_dir: Path, device: str):
        from .nets import skey
        self.skey = skey
        self.device = _device(device)
        self.model, self.sample_rate = skey.load_checkpoint(model_store.file_path(models_dir, spec), self.device)
        # Map the classifier's output order onto Key.index.
        self.order = [Key.parse(name).index for name in skey.KEY_NAMES]

    def key_probs(self, y: np.ndarray, tuning_cents: float) -> np.ndarray:
        import torch
        if abs(tuning_cents) > 5:
            import soxr  # retune by resampling (the tempo change is irrelevant for the key)
            y = soxr.resample(y, SAMPLE_RATE, SAMPLE_RATE * rhythm.semitone_ratio(tuning_cents))
        if self.sample_rate != SAMPLE_RATE:
            import soxr
            y = soxr.resample(y, SAMPLE_RATE, self.sample_rate)
        audio = torch.tensor(audio_io.peak_normalize(np.asarray(y, np.float32)), device=self.device)
        with torch.inference_mode():
            p = self.model(audio).float().cpu().numpy()
        probs = np.zeros(24)
        probs[self.order] = p
        return probs / probs.sum()

    def close(self) -> None:
        del self.model


_BACKENDS = {models.BTC: ChordBackend, models.BEAT_THIS: BeatThisBackend,
             models.LIBROSA_BEATS: LibrosaBeatBackend, models.SKEY: SKeyBackend}


def estimate_tuning(y: np.ndarray) -> float:
    """Deviation of the song's reference pitch from A = 440 Hz, in cents (−50…+50)."""
    import librosa
    # Analyse at most ~3 minutes from the middle; more adds time but not precision.
    limit = 180 * SAMPLE_RATE
    if len(y) > limit:
        start = (len(y) - limit) // 2
        y = y[start:start + limit]
    if not np.any(np.abs(y) > 1e-4):
        return 0.0
    tuning = librosa.estimate_tuning(y=y, sr=SAMPLE_RATE, bins_per_octave=12, resolution=0.005)
    return float(tuning) * 100.0


def chroma_from_cqt(magnitude: np.ndarray) -> np.ndarray:
    """Fold a 24-bins-per-octave CQT (lowest bin = C) into 12 pitch classes."""
    energy = (magnitude ** 2).mean(axis=0)
    chroma = np.zeros(12)
    for i, e in enumerate(energy):
        chroma[(i // 2) % 12] += e
    return np.sqrt(chroma)


# ================================================================== pipeline
class Pipeline:
    """Keeps loaded models between songs so a queue doesn't reload them every time."""

    def __init__(self) -> None:
        self._loaded: dict[str, tuple[tuple, object]] = {}

    def backend(self, task: str, s: Settings, callbacks: Callbacks):
        spec = s.spec(task)
        key = (spec.id, s.device, str(s.models_dir))
        current = self._loaded.get(task)
        if current and current[0] == key:
            return current[1]
        self.release(task)
        if spec.backend == models.PROFILE_KEY:
            return None
        ensure_model(spec, s, callbacks)
        if spec.needs_download:
            callbacks.log(f"Loading {spec.label}…")
        backend = _BACKENDS[spec.backend](spec, s.models_dir, s.device)
        self._loaded[task] = (key, backend)
        return backend

    def release(self, task: str | None = None) -> None:
        for t in [task] if task else list(self._loaded):
            entry = self._loaded.pop(t, None)
            if entry:
                entry[1].close()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def analyze(self, path: Path, s: Settings, callbacks: Callbacks | None = None) -> Analysis:
        cb = callbacks or Callbacks()
        path = Path(path)
        if s.use_cache:
            cached = load_cached(path, s)
            if cached:
                cb.log(f"{path.name}: using the earlier result.")
                cb.progress(1.0, "Done")
                if s.save_chart:
                    write_chart(path, cached)
                return cached

        def check() -> None:
            if cb.is_cancelled():
                raise Cancelled()

        def stage(name: str, lo: float, hi: float) -> Callable[[float], None]:
            cb.progress(lo, name)
            return lambda f: cb.progress(lo + (hi - lo) * max(0.0, min(1.0, f)), name)

        started = time.perf_counter()
        # Load models first, so a missing download fails before the slow work starts.
        stage("Loading models", 0.0, 0.05)
        chord_backend = self.backend(models.CHORDS, s, cb)
        beat_backend = self.backend(models.BEATS, s, cb)
        key_backend = self.backend(models.KEY, s, cb)
        check()

        stage("Decoding", 0.05, 0.1)
        y = audio_io.load_mono(path, SAMPLE_RATE)
        duration = len(y) / SAMPLE_RATE
        if duration < 2:
            raise ValueError("The file is too short to analyse (under 2 seconds).")
        check()

        stage("Tuning", 0.1, 0.15)
        cents = estimate_tuning(y)
        applied = cents if s.tuning_correction else 0.0
        check()

        stage("Beats", 0.15, 0.4)
        beats, downbeats = beat_backend.track(y)
        tempo = rhythm.tempo_from_beats(beats)
        meter = rhythm.meter_from_downbeats(beats, downbeats)
        check()

        chord_progress = stage("Chords", 0.4, 0.85)
        magnitude = chord_backend.features(y, applied)
        chord_progress(0.3)
        frame_labels = chord_backend.predict(magnitude, cb.is_cancelled,
                                             lambda f: chord_progress(0.3 + 0.7 * f))
        chords = segments.frames_to_segments(frame_labels, chord_backend.btc.FRAME_SECONDS, duration)
        chords = [(st, en, "N" if lab == "X" else lab) for st, en, lab in chords]
        chords = segments.remove_short(chords, s.min_chord_seconds)
        if s.snap_to_beats and len(beats) > 4:
            chords = segments.snap_to_beats(chords, beats)
            chords = segments.remove_short(chords, s.min_chord_seconds)
        check()

        stage("Key", 0.85, 0.97)
        if key_backend is not None:
            probs = key_backend.key_probs(y, applied)
        else:
            _, probs = theory.estimate_key(chroma_from_cqt(magnitude), chords)
        key_index = int(np.argmax(probs))

        result = Analysis(
            path=str(path), duration=round(duration, 3),
            bpm=round(tempo.bpm, 2), bpm_low=round(tempo.low, 2), bpm_high=round(tempo.high, 2),
            bpm_stable=bool(tempo.stable), meter=meter,
            beats=[round(float(b), 3) for b in beats], downbeats=[round(float(d), 3) for d in downbeats],
            tuning_cents=round(cents, 2), key_index=key_index, key_probs=[round(float(p), 4) for p in probs],
            chords=[(round(st, 3), round(en, 3), lab) for st, en, lab in chords],
            models={spec.task: spec.id for spec in s.specs()},
            options={"snap_to_beats": s.snap_to_beats, "min_chord_seconds": s.min_chord_seconds,
                     "tuning_correction": s.tuning_correction},
            elapsed=round(time.perf_counter() - started, 2),
        )
        cb.progress(1.0, "Done")
        key = result.key
        cb.log(f"{path.name}: {result.tempo.text()}, {key.name} ({key.camelot}), "
               f"{rhythm.describe_tuning(cents)}, {len(result.chords)} chord changes "
               f"({result.elapsed:.1f} s).")
        if s.use_cache:
            store_cached(path, s, result)
        if s.save_chart:
            write_chart(path, result)
        return result


def chart_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).stem + ".chords.txt")


def write_chart(path: Path, result: Analysis) -> None:
    chart_path(path).write_text(result.chart(), encoding="utf-8")


def ensure_model(spec: ModelSpec, s: Settings, callbacks: Callbacks) -> None:
    """Download a model if needed (and allowed)."""
    if model_store.is_installed(s.models_dir, spec):
        return
    if not s.allow_download:
        raise RuntimeError(f"{spec.label} is not downloaded and model downloads are disabled "
                           "(Models & performance).")
    model_store.download(s.models_dir, spec, callbacks.log, lambda f: callbacks.progress(f, "Downloading"),
                         callbacks.is_cancelled)
