"""Music theory: keys, Camelot codes, chord symbols and harmonic functions.

Pitch classes are integers, C = 0 … B = 11. Chords come from the chord model
as Harte-style labels ("C", "A:min", "F#:hdim7", "N" for no chord). Roman
numerals are written relative to the major scale of the tonic in both modes,
the convention used in pop and jazz analysis: in A minor, F major is bVI and
G major is bVII.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
MAJOR, MINOR = "major", "minor"

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
HARMONIC_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 11)

# Conventional key names: the spelling with the fewest accidentals.
_MAJOR_TONIC = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_TONIC = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
_FLAT_MAJOR_KEYS = {5, 10, 3, 8, 1}          # F Bb Eb Ab Db
_FLAT_MINOR_KEYS = {2, 7, 0, 5, 10, 3}       # Dm Gm Cm Fm Bbm Ebm

_NOTE_TO_PC = {name: pc for names in (SHARP_NAMES, FLAT_NAMES) for pc, name in enumerate(names)}
_NOTE_TO_PC.update({"Cb": 11, "B#": 0, "E#": 5, "Fb": 4})


def note_to_pc(name: str) -> int:
    try:
        return _NOTE_TO_PC[name.strip()]
    except KeyError:
        raise ValueError(f"Unknown note name: {name!r}") from None


# ===================================================================== keys
@dataclass(frozen=True)
class Key:
    tonic: int   # pitch class
    mode: str    # MAJOR or MINOR

    @property
    def tonic_name(self) -> str:
        return (_MAJOR_TONIC if self.mode == MAJOR else _MINOR_TONIC)[self.tonic]

    @property
    def name(self) -> str:
        """e.g. "Eb major", "F# minor"."""
        return f"{self.tonic_name} {self.mode}"

    @property
    def short(self) -> str:
        """e.g. "Eb", "F#m"."""
        return self.tonic_name + ("m" if self.mode == MINOR else "")

    @property
    def uses_flats(self) -> bool:
        return self.tonic in (_FLAT_MAJOR_KEYS if self.mode == MAJOR else _FLAT_MINOR_KEYS)

    @property
    def camelot(self) -> str:
        """Camelot wheel code: C major = 8B, A minor = 8A, G major = 9B…"""
        major_tonic = self.tonic if self.mode == MAJOR else (self.tonic + 3) % 12
        number = (major_tonic * 7 + 7) % 12 + 1  # one step per fifth, C = 8
        return f"{number}{'B' if self.mode == MAJOR else 'A'}"

    @property
    def open_key(self) -> str:
        """Open Key notation (Traktor): C major = 1d, A minor = 1m."""
        number = (int(self.camelot[:-1]) + 4) % 12 + 1
        return f"{number}{'d' if self.mode == MAJOR else 'm'}"

    @property
    def relative(self) -> "Key":
        if self.mode == MAJOR:
            return Key((self.tonic + 9) % 12, MINOR)
        return Key((self.tonic + 3) % 12, MAJOR)

    @property
    def index(self) -> int:
        """0–11 major keys, 12–23 minor keys (by tonic pitch class)."""
        return self.tonic + (12 if self.mode == MINOR else 0)

    @staticmethod
    def from_index(i: int) -> "Key":
        return Key(i % 12, MINOR if i >= 12 else MAJOR)

    @staticmethod
    def parse(text: str) -> "Key":
        """Parse "C major", "F# minor", "Ebm", "Bb"."""
        t = text.strip()
        lower = t.lower()
        if lower.endswith(" minor") or lower.endswith(" min"):
            return Key(note_to_pc(t.rsplit(" ", 1)[0]), MINOR)
        if lower.endswith(" major") or lower.endswith(" maj"):
            return Key(note_to_pc(t.rsplit(" ", 1)[0]), MAJOR)
        if t.endswith("m") and len(t) > 1:
            return Key(note_to_pc(t[:-1]), MINOR)
        return Key(note_to_pc(t), MAJOR)

    def spell(self, pc: int) -> str:
        """Name a pitch class with this key's accidentals."""
        return (FLAT_NAMES if self.uses_flats else SHARP_NAMES)[pc % 12]

    def scale(self) -> tuple[int, ...]:
        steps = MAJOR_SCALE if self.mode == MAJOR else NATURAL_MINOR_SCALE
        return tuple((self.tonic + s) % 12 for s in steps)

    def __str__(self) -> str:
        return self.name


ALL_KEYS = [Key.from_index(i) for i in range(24)]


# =================================================================== chords
@dataclass(frozen=True)
class Quality:
    intervals: tuple[int, ...]
    symbol: str        # appended to the root name: "m7" → "Am7"
    minor_case: bool   # Roman numeral in lower case
    numeral: str       # appended to the Roman numeral: "ø7" → "viiø7"
    triad: str         # the quality this simplifies to


QUALITIES: dict[str, Quality] = {
    "maj": Quality((0, 4, 7), "", False, "", "maj"),
    "min": Quality((0, 3, 7), "m", True, "", "min"),
    "dim": Quality((0, 3, 6), "dim", True, "°", "dim"),
    "aug": Quality((0, 4, 8), "aug", False, "+", "aug"),
    "maj6": Quality((0, 4, 7, 9), "6", False, "6", "maj"),
    "min6": Quality((0, 3, 7, 9), "m6", True, "6", "min"),
    "7": Quality((0, 4, 7, 10), "7", False, "7", "maj"),
    "maj7": Quality((0, 4, 7, 11), "maj7", False, "maj7", "maj"),
    "min7": Quality((0, 3, 7, 10), "m7", True, "7", "min"),
    "minmaj7": Quality((0, 3, 7, 11), "m(maj7)", True, "(maj7)", "min"),
    "dim7": Quality((0, 3, 6, 9), "dim7", True, "°7", "dim"),
    "hdim7": Quality((0, 3, 6, 10), "m7b5", True, "ø7", "dim"),
    "sus2": Quality((0, 2, 7), "sus2", False, "sus2", "sus2"),
    "sus4": Quality((0, 5, 7), "sus4", False, "sus4", "sus4"),
}
_DOMINANTS = {"maj", "7"}
_LEADING_TONE = {"dim", "dim7", "hdim7"}


@dataclass(frozen=True)
class Chord:
    root: int | None   # None = no chord
    quality: str = "maj"

    @staticmethod
    def parse(label: str) -> "Chord":
        """Parse a Harte-style label: "C", "C:min", "Bb:maj7", "N", "X"."""
        label = label.strip()
        if label in ("N", "X", ""):
            return NO_CHORD
        root, _, quality = label.partition(":")
        quality = quality.split("/")[0] or "maj"
        if quality not in QUALITIES:
            raise ValueError(f"Unknown chord quality in {label!r}")
        return Chord(note_to_pc(root), quality)

    @property
    def is_chord(self) -> bool:
        return self.root is not None

    @property
    def label(self) -> str:
        """Harte-style label (sharps), as stored in results."""
        if self.root is None:
            return "N"
        return SHARP_NAMES[self.root] + ("" if self.quality == "maj" else f":{self.quality}")

    def pitch_classes(self) -> tuple[int, ...]:
        if self.root is None:
            return ()
        return tuple((self.root + i) % 12 for i in QUALITIES[self.quality].intervals)

    def simplified(self) -> "Chord":
        if self.root is None:
            return self
        return Chord(self.root, QUALITIES[self.quality].triad)

    def symbol(self, key: Key | None = None) -> str:
        """Chord symbol such as "Fmaj7", "Bbm", "F#m7b5"; spelled to fit the key."""
        if self.root is None:
            return "N.C."
        root = key.spell(self.root) if key else SHARP_NAMES[self.root]
        return root + QUALITIES[self.quality].symbol


NO_CHORD = Chord(None)

# Degree names relative to the major scale of the tonic.
_DEGREES = ("I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII")


def _numeral(interval: int, quality: str) -> str:
    q = QUALITIES[quality]
    degree = _DEGREES[interval % 12]
    if degree == "#IV" and not q.minor_case:
        degree = "bV"  # a major chord a tritone above the tonic is usually heard as bV
    if q.minor_case:
        degree = degree.lower()
    return degree + q.numeral


def is_diatonic(chord: Chord, key: Key) -> bool:
    """All chord tones in the key's scale (minor keys also allow harmonic minor)."""
    pcs = {(p - key.tonic) % 12 for p in chord.pitch_classes()}
    if key.mode == MAJOR:
        return pcs <= set(MAJOR_SCALE)
    return pcs <= set(NATURAL_MINOR_SCALE) or pcs <= set(HARMONIC_MINOR_SCALE)


def _diatonic_triad_on(interval: int, key: Key) -> str | None:
    """Quality of the diatonic triad built on a scale degree, or None if not a degree."""
    scale = MAJOR_SCALE if key.mode == MAJOR else NATURAL_MINOR_SCALE
    if interval not in scale:
        return None
    if key.mode == MINOR and interval == 7:
        return "maj"  # the dominant of a minor key is major (harmonic minor)
    for quality in ("maj", "min", "dim"):
        intervals = {(interval + i) % 12 for i in QUALITIES[quality].intervals}
        if intervals <= set(scale):
            return quality
    return None


def function(chord: Chord, key: Key) -> str:
    """The chord's harmonic function as a Roman numeral, e.g. "IVmaj7", "V7/vi", "bVII"."""
    if chord.root is None:
        return ""
    interval = (chord.root - key.tonic) % 12
    if not is_diatonic(chord, key):
        secondary = _secondary(chord, key)
        if secondary:
            return secondary
    return _numeral(interval, chord.quality)


def _secondary(chord: Chord, key: Key) -> str | None:
    """Secondary dominant (V/x) or leading-tone chord (vii°/x) of a diatonic degree."""
    if chord.quality in _DOMINANTS:
        if chord.quality == "maj" and key.mode == MINOR:
            return None  # in minor, non-diatonic major triads read better as borrowed chords
        target = (chord.root + 5 - key.tonic) % 12   # a fifth below
        numeral = "V" + QUALITIES[chord.quality].numeral
    elif chord.quality in _LEADING_TONE:
        target = (chord.root + 1 - key.tonic) % 12   # a semitone above
        numeral = "vii" + QUALITIES[chord.quality].numeral
    else:
        return None
    if target == 0:
        return None
    target_quality = _diatonic_triad_on(target, key)
    if target_quality not in ("maj", "min"):
        return None
    return f"{numeral}/{_numeral(target, target_quality)}"


def describe(chord: Chord, key: Key | None, with_function: bool = True) -> str:
    """"Fmaj7 (IVmaj7)" – the chord symbol, plus its function when a key is known."""
    symbol = chord.symbol(key)
    if not with_function or key is None or not chord.is_chord:
        return symbol
    return f"{symbol} ({function(chord, key)})"


# ========================================================== key estimation
# Krumhansl-Kessler probe-tone profiles, C major and C minor.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def profile_scores(pc_weights: np.ndarray) -> np.ndarray:
    """Correlation of a 12-bin pitch-class vector with all 24 key profiles.

    Returns an array indexed like Key.index (0–11 major, 12–23 minor).
    """
    v = np.asarray(pc_weights, dtype=float)
    scores = np.zeros(24)
    if not np.any(v > 0) or np.allclose(v, v[0]):
        return scores
    for tonic in range(12):
        scores[tonic] = np.corrcoef(v, np.roll(KK_MAJOR, tonic))[0, 1]
        scores[12 + tonic] = np.corrcoef(v, np.roll(KK_MINOR, tonic))[0, 1]
    return np.nan_to_num(scores)


def chord_pitch_histogram(segments: list[tuple[float, float, str]]) -> np.ndarray:
    """Duration-weighted pitch classes of a chord sequence; roots count double."""
    hist = np.zeros(12)
    for start, end, label in segments:
        chord = Chord.parse(label)
        if not chord.is_chord:
            continue
        dur = max(0.0, end - start)
        for pc in chord.pitch_classes():
            hist[pc] += dur
        hist[chord.root] += dur
    return hist


def tonic_chord_share(segments: list[tuple[float, float, str]]) -> np.ndarray:
    """For each of the 24 keys: share of the song spent on its tonic chord."""
    share = np.zeros(24)
    total = sum(max(0.0, e - s) for s, e, lab in segments if Chord.parse(lab).is_chord) or 1.0
    for start, end, label in segments:
        chord = Chord.parse(label)
        if not chord.is_chord:
            continue
        triad = QUALITIES[chord.quality].triad
        if triad == "maj":
            share[chord.root] += (end - start) / total
        elif triad == "min":
            share[12 + chord.root] += (end - start) / total
    return share


def estimate_key(chroma: np.ndarray | None, segments: list[tuple[float, float, str]]) -> tuple[Key, np.ndarray]:
    """Key from an audio chroma vector and/or a chord sequence.

    Returns the best key and a probability-like score for all 24 keys.
    """
    combined = np.zeros(24)
    parts = 0
    if chroma is not None and np.any(np.asarray(chroma) > 0):
        combined += profile_scores(chroma)
        parts += 1
    if segments:
        combined += profile_scores(chord_pitch_histogram(segments))
        parts += 1
        # Songs tend to start and end on the tonic and return to it often.
        combined += 0.5 * tonic_chord_share(segments)
    if not parts:
        return Key(0, MAJOR), np.full(24, 1 / 24)
    probs = np.exp(8.0 * (combined - combined.max()))
    probs /= probs.sum()
    return Key.from_index(int(np.argmax(probs))), probs
