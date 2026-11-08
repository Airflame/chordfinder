"""Decoding audio (and the audio track of videos) for analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".aiff", ".aif",
                    ".mp4", ".mkv", ".webm", ".mov"}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def load_mono(path: Path, sample_rate: int, max_seconds: float | None = None) -> np.ndarray:
    """Decode any audio/video file to mono float32 at sample_rate (via PyAV's bundled FFmpeg)."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    limit = int(max_seconds * sample_rate) if max_seconds else None
    have = 0
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"{Path(path).name} has no audio track.")
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunk = out.to_ndarray().reshape(-1)
                chunks.append(chunk)
                have += len(chunk)
            if limit and have >= limit:
                break
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    return audio[:limit] if limit else audio


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio / peak if peak > 1e-6 else audio


def collect(paths: list[Path]) -> list[Path]:
    """Expand folders into the audio files they contain (recursively, sorted)."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and is_audio(f)))
        elif p.is_file() and is_audio(p):
            out.append(p)
    return out
