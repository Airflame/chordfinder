import numpy as np
import pytest

from chordfinder.theory import ALL_KEYS, Chord, Key, MAJOR, MINOR, describe, estimate_key, function

C, A_MIN = Key.parse("C major"), Key.parse("A minor")


@pytest.mark.parametrize("key, code", [
    ("C major", "8B"), ("A minor", "8A"), ("G major", "9B"), ("E minor", "9A"), ("F major", "7B"),
    ("D minor", "7A"), ("B major", "1B"), ("G# minor", "1A"), ("F# major", "2B"), ("Eb minor", "2A"),
    ("Db major", "3B"), ("Bb minor", "3A"), ("Ab major", "4B"), ("Eb major", "5B"), ("C minor", "5A"),
    ("Bb major", "6B"), ("G minor", "6A"), ("D major", "10B"), ("B minor", "10A"), ("A major", "11B"),
    ("F# minor", "11A"), ("E major", "12B"), ("C# minor", "12A"),
])
def test_camelot(key, code):
    assert Key.parse(key).camelot == code


def test_camelot_is_a_bijection_and_relatives_share_a_number():
    codes = {k.camelot for k in ALL_KEYS}
    assert len(codes) == 24
    for k in ALL_KEYS:
        assert k.camelot[:-1] == k.relative.camelot[:-1]


def test_open_key():
    assert C.open_key == "1d" and A_MIN.open_key == "1m" and Key.parse("G").open_key == "2d"


@pytest.mark.parametrize("text, name", [
    ("Eb", "Eb major"), ("D#m", "Eb minor"), ("C# major", "Db major"), ("g# minor", "G# minor"),
    ("A#", "Bb major"), ("Bb minor", "Bb minor"),
])
def test_key_names_use_conventional_spelling(text, name):
    assert Key.parse(text.capitalize() if text[0].islower() else text).name == name


def test_index_round_trip():
    for i in range(24):
        assert Key.from_index(i).index == i


@pytest.mark.parametrize("label, key, symbol", [
    ("F:maj7", C, "Fmaj7"), ("A#", Key.parse("F"), "Bb"), ("A#:min", Key.parse("F#"), "A#m"),
    ("B:hdim7", C, "Bm7b5"), ("C#:dim7", C, "C#dim7"), ("D:sus4", C, "Dsus4"), ("G:7", C, "G7"),
    ("E:minmaj7", C, "Em(maj7)"), ("N", C, "N.C."),
])
def test_chord_symbols(label, key, symbol):
    assert Chord.parse(label).symbol(key) == symbol


def test_describe_matches_the_classic_example():
    assert describe(Chord.parse("F:maj7"), C) == "Fmaj7 (IVmaj7)"


@pytest.mark.parametrize("label, expected", [
    ("C", "I"), ("D:min7", "ii7"), ("E:min", "iii"), ("F:maj7", "IVmaj7"), ("G:7", "V7"), ("A:min", "vi"),
    ("B:dim", "vii°"), ("B:hdim7", "viiø7"), ("G:sus4", "Vsus4"), ("F:maj6", "IV6"),
    # secondary dominants and leading-tone chords
    ("D", "V/V"), ("D:7", "V7/V"), ("E:7", "V7/vi"), ("A", "V/ii"), ("C:7", "V7/IV"), ("B:7", "V7/iii"),
    ("F#:dim7", "vii°7/V"), ("C#:dim7", "vii°7/ii"), ("G#:dim", "vii°/vi"),
    # borrowed and chromatic chords
    ("A#", "bVII"), ("G#", "bVI"), ("D#", "bIII"), ("F:min", "iv"), ("C#", "bII"), ("F#:min7", "#iv7"),
    ("A#:7", "bVII7"),
])
def test_functions_in_c_major(label, expected):
    assert function(Chord.parse(label), C) == expected


@pytest.mark.parametrize("label, expected", [
    ("A:min", "i"), ("B:dim", "ii°"), ("C", "bIII"), ("D:min", "iv"), ("E:min", "v"), ("E", "V"),
    ("E:7", "V7"), ("F", "bVI"), ("G", "bVII"), ("G#:dim7", "vii°7"), ("D", "IV"), ("A", "I"),
    ("B:7", "V7/V"), ("A:7", "V7/iv"),
])
def test_functions_in_a_minor(label, expected):
    assert function(Chord.parse(label), A_MIN) == expected


def test_simplified_chords():
    assert Chord.parse("F:maj7").simplified() == Chord.parse("F")
    assert Chord.parse("B:hdim7").simplified() == Chord.parse("B:dim")
    assert Chord.parse("D:sus4").simplified() == Chord.parse("D:sus4")


def test_bad_labels_raise():
    with pytest.raises(ValueError):
        Chord.parse("H:min")
    with pytest.raises(ValueError):
        Chord.parse("C:weird")


def _progression(labels, seconds=2.0, repeats=4):
    out, t = [], 0.0
    for _ in range(repeats):
        for lab in labels:
            out.append((t, t + seconds, lab))
            t += seconds
    return out


@pytest.mark.parametrize("labels, key", [
    (["C", "F", "G", "C"], "C major"),
    (["A:min", "D:min", "E", "A:min"], "A minor"),
    (["G", "D", "E:min", "C"], "G major"),
    (["D#", "G#", "A#:7", "D#"], "Eb major"),
    (["F#:min", "B:min", "C#:7", "F#:min"], "F# minor"),
    # vi-IV-I-V starts on the minor chord but is heard in the major key
    (["F#:min", "D", "A", "E"], "A major"),
])
def test_estimate_key_from_chords(labels, key):
    best, probs = estimate_key(None, _progression(labels))
    assert best == Key.parse(key)
    assert probs.shape == (24,) and abs(probs.sum() - 1) < 1e-9


def test_estimate_key_from_chroma_only():
    chroma = np.zeros(12)
    for pc in (0, 2, 4, 5, 7, 9, 11):
        chroma[pc] = 1.0
    chroma[0] = chroma[7] = 2.0
    best, _ = estimate_key(chroma, [])
    assert best == Key(0, MAJOR)


def test_estimate_key_without_information():
    best, probs = estimate_key(None, [(0, 5, "N")])
    assert best.mode in (MAJOR, MINOR) and np.allclose(probs, 1 / 24)
