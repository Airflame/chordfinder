"""Analysis of synthesised songs with known tempo, key, tuning and chords."""

import numpy as np
import pytest

from chordfinder import analyzer, model_store, models, rhythm
from chordfinder.analyzer import SAMPLE_RATE, Analysis, Pipeline, Settings
from chordfinder.theory import Chord, Key

PROGRESSION = ["C", "A:min", "F", "G"]  # I vi IV V in C major


def tone(freq, seconds, sr=SAMPLE_RATE):
    t = np.arange(int(seconds * sr)) / sr
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h for h in range(1, 6))
    env = np.minimum(1, t * 50) * np.exp(-t * 0.6)
    return wave * env


def song(a4=440.0, bpm=120.0, bars=16, sr=SAMPLE_RATE):
    """One chord per 4/4 bar, triads in the middle register, a bass note and a kick on each beat."""
    beat = 60.0 / bpm
    bar = 4 * beat
    out = np.zeros(int(bars * bar * sr) + sr)
    midi = lambda n: a4 * 2 ** ((n - 69) / 12)
    for i in range(bars):
        chord = Chord.parse(PROGRESSION[i % len(PROGRESSION)])
        start = int(i * bar * sr)
        notes = sorted(60 + ((pc - 0) % 12) for pc in chord.pitch_classes())
        layer = sum(tone(midi(n), bar) for n in notes) + 1.5 * tone(midi(36 + chord.root), bar)
        out[start:start + len(layer)] += layer
        for b in range(4):
            k = int((i * bar + b * beat) * sr)
            n = int(0.08 * sr)
            kick = np.sin(2 * np.pi * 55 * np.arange(n) / sr) * np.exp(-np.arange(n) / (0.02 * sr))
            out[k:k + n] += 3 * kick
    return (out / np.abs(out).max() * 0.8).astype(np.float32)


@pytest.mark.parametrize("a4", [440.0, 432.0, 446.0])
def test_tuning_detection(a4):
    cents = analyzer.estimate_tuning(song(a4, bars=8))
    assert abs(rhythm.cents_to_a4(cents) - a4) < 1.0


def test_chroma_from_cqt_folds_octaves():
    mag = np.zeros((3, 144))
    mag[:, 0] = mag[:, 24] = 1.0     # C in two octaves
    mag[:, 14] = 1.0                 # G
    chroma = analyzer.chroma_from_cqt(mag)
    assert np.argmax(chroma) == 0 and chroma[7] > 0 and chroma[1] == 0


def test_result_json_round_trip_and_exports():
    r = Analysis(path="x.mp3", duration=10, bpm=120, bpm_low=119, bpm_high=121, bpm_stable=True, meter=4,
                 beats=[0.5, 1.0], downbeats=[0.5], tuning_cents=-31.8, key_index=0,
                 key_probs=[1.0] + [0.0] * 23, chords=[(0.0, 2.0, "C"), (2.0, 4.0, "F:maj7"), (4.0, 5.0, "N")],
                 models={}, options={})
    again = Analysis.from_json(r.to_json())
    assert again == r and again.key == Key(0, "major") and abs(again.a4_hz - 432) < 0.1
    assert "0:02.00  Fmaj7 (IVmaj7)" in r.chart()
    assert "N.C." not in r.chart()
    assert r.lab().splitlines()[1] == "2.000\t4.000\tF:maj7"


def _installed(*ids):
    return all(model_store.is_installed(analyzer.default_models_dir(), models.get(i)) for i in ids)


@pytest.mark.skipif(not _installed("btc-large-voca", "beat-this-final0", "skey"),
                    reason="models not downloaded")
@pytest.mark.parametrize("key_model, beat_model", [("skey", "beat-this-final0"), ("profile-key", "librosa-beats")])
def test_end_to_end(tmp_path, key_model, beat_model):
    import soundfile as sf
    path = tmp_path / "song.wav"
    sf.write(path, song(a4=432.0, bpm=120.0, bars=16), SAMPLE_RATE)
    s = Settings(key_model=key_model, beat_model=beat_model, cache_dir=tmp_path / "cache", allow_download=False)
    r = Pipeline().analyze(path, s)
    if beat_model == "librosa-beats":
        # The classic tracker may lock onto another metrical level (here the strong chord onsets per bar).
        assert any(abs(r.bpm - 120 * f) < 1.0 for f in (0.25, 0.5, 1, 2))
    else:
        assert abs(r.bpm - 120) < 1.0 and r.meter == 4
    assert abs(r.a4_hz - 432) < 1.5
    assert r.key == Key(0, "major")
    # Most of the song is labelled with the right chord (sevenths/sixths of the right triad also count).
    bar = 2.0
    right = 0.0
    for start, end, label in r.chords:
        chord = Chord.parse(label).simplified()
        for i in range(16):
            lo, hi = max(start, i * bar), min(end, (i + 1) * bar)
            if hi > lo and chord == Chord.parse(PROGRESSION[i % 4]):
                right += hi - lo
    assert right / (16 * bar) > 0.8
    # The second run comes from the cache.
    assert Pipeline().analyze(path, s).chords == r.chords
