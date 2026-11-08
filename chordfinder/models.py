"""The analysis models the app can download and run.

Every model is a small PyTorch checkpoint published by its authors. Each file
is pinned by its SHA-256 hash, so a download that was tampered with or cut
short is never loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tasks (one model per task is active at a time)
CHORDS, BEATS, KEY = "chords", "beats", "key"
TASK_LABELS = {CHORDS: "Chord model", BEATS: "Beat model", KEY: "Key model"}

# Backends
BTC, BEAT_THIS, SKEY = "btc", "beat-this", "skey"
LIBROSA_BEATS, PROFILE_KEY = "librosa-beats", "profile-key"


@dataclass(frozen=True)
class ModelFile:
    name: str      # file name inside the model's folder
    url: str
    sha256: str
    size: int      # bytes


@dataclass(frozen=True)
class ModelSpec:
    id: str            # stable identifier, also the folder name
    label: str
    task: str          # CHORDS, BEATS or KEY
    backend: str
    files: tuple[ModelFile, ...] = ()
    note: str = ""
    params: tuple = ()  # backend-specific settings as (key, value) pairs

    @property
    def subdir(self) -> str:
        """Subfolder of the models folder that holds this task's models."""
        return self.task

    @property
    def size_mb(self) -> float:
        return sum(f.size for f in self.files) / 2**20

    @property
    def needs_download(self) -> bool:
        return bool(self.files)

    def param(self, key: str, default=None):
        return dict(self.params).get(key, default)


_BTC_URL = "https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/master/test/"
_BEAT_THIS_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/"
_SKEY_URL = "https://raw.githubusercontent.com/deezer/skey/main/skey/models/"

MODELS: list[ModelSpec] = [
    # ------------------------------------------------------------ chords
    ModelSpec(
        "btc-large-voca", "BTC, 170 chord types (recommended)", CHORDS, BTC,
        (ModelFile("btc_model_large_voca.pt", _BTC_URL + "btc_model_large_voca.pt",
                   "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51", 12229576),),
        "Bi-directional Transformer for Chord recognition (Park et al., ISMIR 2019). Recognises major, minor, "
        "dim, aug, 6, 7, maj7, m7, mMaj7, dim7, m7b5, sus2 and sus4 chords on all 12 roots.",
        (("num_chords", 170),),
    ),
    ModelSpec(
        "btc-majmin", "BTC, major and minor only", CHORDS, BTC,
        (ModelFile("btc_model.pt", _BTC_URL + "btc_model.pt",
                   "71c2c5db17e8c43b8a9a9da5db36ef2d667158c07a214eba16344c154c00bf54", 12154754),),
        "The same network trained on 24 major/minor triads only. Simpler, steadier chord charts.",
        (("num_chords", 25),),
    ),
    # ------------------------------------------------------------- beats
    ModelSpec(
        "beat-this-final0", "Beat This! (recommended)", BEATS, BEAT_THIS,
        (ModelFile("final0.ckpt", _BEAT_THIS_URL + "final0.ckpt",
                   "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331", 81058141),),
        "State-of-the-art beat and downbeat tracker (Foscarin et al., ISMIR 2024). Also finds the bar lines.",
    ),
    ModelSpec(
        "beat-this-small0", "Beat This! small (fast)", BEATS, BEAT_THIS,
        (ModelFile("small0.ckpt", _BEAT_THIS_URL + "small0.ckpt",
                   "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b", 8451101),),
        "A 10× smaller Beat This! model, slightly less accurate.",
    ),
    ModelSpec(
        "librosa-beats", "Classic beat tracker (no download)", BEATS, LIBROSA_BEATS, (),
        "librosa's dynamic-programming beat tracker. No neural network, no bar lines; "
        "less reliable on syncopated music.",
    ),
    # --------------------------------------------------------------- key
    ModelSpec(
        "skey", "S-KEY (recommended)", KEY, SKEY,
        (ModelFile("skey.pt", _SKEY_URL + "skey.pt",
                   "78dfd0ad4fa9434bf7cec70a25934b7c575bda9c80e994700140770ad3a5ead4", 765465),),
        "Self-supervised key detector from Deezer Research (Kong et al., ICASSP 2025).",
    ),
    ModelSpec(
        "profile-key", "Chord and chroma profiles (no download)", KEY, PROFILE_KEY, (),
        "Matches the song's pitch-class content and its recognised chords against major and minor key "
        "profiles (Krumhansl-Kessler).",
    ),
]

_BY_ID = {m.id: m for m in MODELS}
DEFAULTS = {CHORDS: "btc-large-voca", BEATS: "beat-this-final0", KEY: "skey"}


def for_task(task: str) -> list[ModelSpec]:
    return [m for m in MODELS if m.task == task]


def get(model_id: str, task: str | None = None) -> ModelSpec:
    """Look up a model; unknown ids fall back to the task's default."""
    spec = _BY_ID.get(model_id)
    if spec is not None and (task is None or spec.task == task):
        return spec
    return _BY_ID[DEFAULTS[task or CHORDS]]
