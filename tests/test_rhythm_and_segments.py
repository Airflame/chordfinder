import numpy as np
import pytest

from chordfinder import rhythm
from chordfinder.segments import frames_to_segments, merge_neighbours, remove_short, segment_at, snap_to_beats


def beat_grid(bpm, n=200, start=0.5, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = start + np.arange(n) * 60.0 / bpm
    return t + rng.normal(0, jitter, n) if jitter else t


@pytest.mark.parametrize("bpm", [70.0, 98.5, 120.0, 128.0, 174.0])
def test_steady_tempo_is_precise(bpm):
    t = rhythm.tempo_from_beats(np.round(beat_grid(bpm, jitter=0.008), 2))  # 10 ms frames, like the model
    assert t.stable and abs(t.bpm - bpm) < 0.1


def test_missed_and_doubled_beats_do_not_change_the_tempo():
    b = list(beat_grid(120))
    del b[50], b[90]                                    # missed beats
    b += [x + 0.25 for x in b[100:140]]                 # a double-time passage
    t = rhythm.tempo_from_beats(sorted(b))
    assert t.stable and abs(t.bpm - 120) < 0.2


def test_tempo_change_is_reported_as_variable():
    b = np.concatenate([beat_grid(100, 100), beat_grid(130, 100, start=60.5)])
    t = rhythm.tempo_from_beats(b)
    assert not t.stable and t.low < 105 and t.high > 125
    assert t.text().startswith("≈")


def test_too_few_beats():
    assert rhythm.tempo_from_beats([0.5, 1.0]).bpm == 0
    assert rhythm.tempo_from_beats([]).text() == "–"


def test_format_bpm():
    assert rhythm.format_bpm(127.96) == "128"
    assert rhythm.format_bpm(127.6) == "127.6"


@pytest.mark.parametrize("meter", [3, 4, 5, 7])
def test_meter(meter):
    beats = beat_grid(120, 96)
    assert rhythm.meter_from_downbeats(beats, beats[::meter]) == meter


def test_meter_unknown_without_downbeats():
    assert rhythm.meter_from_downbeats(beat_grid(120, 20), []) is None


def test_bar_lengths_and_meter_text():
    beats = beat_grid(120, 60)
    downbeats = np.concatenate([beats[0:28:7], beats[28::4]])   # four bars of 7, then eight of 4
    assert rhythm.bar_lengths(beats, downbeats) == [7] * 4 + [4] * 7
    assert rhythm.meters(beats, downbeats) == [4, 7]
    assert rhythm.meter_text([4, 7]) == "4/4 and 7/4" and rhythm.meter_text([5]) == "5/4"
    assert rhythm.meter_text([]) == "–"


def test_a_few_odd_bars_are_not_a_second_meter():
    beats = beat_grid(120, 200)
    downbeats = np.concatenate([beats[0:120:4], beats[123::4]])   # one bar of 3 among 4/4
    assert rhythm.meters(beats, downbeats) == [4]


def split_bars(groups, bars=24, strong=0.95, weak=0.7, seed=0):
    """Beats of a song in sum(groups)/4 as a tracker hears it: every group start looks like a downbeat."""
    rng = np.random.default_rng(seed)
    n = sum(groups)
    beats = beat_grid(132, bars * n)
    strength = rng.uniform(0.0, 0.25, beats.size)
    starts = np.cumsum([0] + list(groups))[:-1]
    for g in starts:
        strength[g::n] = (strong if g == 0 else weak) + rng.uniform(-0.05, 0.05, bars)
    detected = beats[np.isin(np.arange(beats.size) % n, starts)]
    return beats, detected, strength


@pytest.mark.parametrize("groups", [(3, 2), (2, 3), (4, 3), (2, 2, 3), (3, 3, 2, 3)])
def test_split_bars_of_odd_meters_are_joined(groups):
    n = sum(groups)
    beats, detected, strength = split_bars(groups)
    assert rhythm.meter_from_downbeats(beats, detected) != n
    downbeats = rhythm.regroup_downbeats(beats, detected, strength)
    assert rhythm.meter_from_downbeats(beats, downbeats) == n
    assert np.allclose(downbeats, beats[::n])              # and the bars start on the strongest beat


def test_odd_meter_without_strengths():
    beats, detected, _ = split_bars((4, 3))
    downbeats = rhythm.regroup_downbeats(beats, detected)
    assert rhythm.meter_from_downbeats(beats, downbeats) == 7


def test_regroup_keeps_a_passage_in_another_meter():
    beats = beat_grid(132, 7 * 12 + 4 * 12)
    index = np.concatenate([np.arange(0, 84, 7), np.arange(84, 132, 4)])   # 12 bars of 7/4, 12 of 4/4
    strength = np.full(beats.size, 0.1)
    strength[index] = 0.95
    strength[index[:12] + 4] = 0.7                                           # the 7/4 bars are heard as 4 + 3
    detected = beats[np.sort(np.concatenate([index, index[:12] + 4]))]
    downbeats = rhythm.regroup_downbeats(beats, detected, strength)
    assert np.allclose(downbeats, beats[index])
    assert rhythm.meters(beats, downbeats) == [7, 4]


def test_max_meter_limits_the_bar_length():
    beats, detected, strength = split_bars((4, 3))
    downbeats = rhythm.regroup_downbeats(beats, detected, strength, max_meter=4)
    assert max(rhythm.bar_lengths(beats, downbeats)) <= 4
    # Regular bars that are too long are split on the strongest beats.
    beats = beat_grid(120, 160)
    strength = np.full(beats.size, 0.1)
    strength[::4] = 0.8
    downbeats = rhythm.regroup_downbeats(beats, beats[::8], strength, max_meter=4)
    assert rhythm.meter_from_downbeats(beats, downbeats, 4) == 4 and np.allclose(downbeats[:5], beats[:20:4])
    assert rhythm.meters(beats, beats[::8], max_meter=4) == []


def test_regroup_leaves_regular_songs_alone():
    beats = beat_grid(120, 200)
    downbeats = np.delete(beats[::4], [10, 30])            # two missed downbeats
    assert rhythm.regroup_downbeats(beats, downbeats) is not None
    assert np.array_equal(rhythm.regroup_downbeats(beats, downbeats), downbeats)
    assert rhythm.regroup_downbeats(beats, []).size == 0
    assert rhythm.regroup_downbeats([], []).size == 0


def test_bar_position():
    beats = beat_grid(120, 16, start=0.0)       # a beat every 0.5 s
    downbeats = beats[::4]
    assert rhythm.bar_position(0.1, beats, downbeats) == (1, 1)
    assert rhythm.bar_position(1.6, beats, downbeats) == (1, 4)
    assert rhythm.bar_position(2.0, beats, downbeats) == (2, 1)


@pytest.mark.parametrize("cents, text", [
    (0.0, "A = 440.0 Hz (standard)"), (-31.8, "A = 432.0 Hz (−32 ¢)"), (15.7, "A = 444.0 Hz (+16 ¢)"),
])
def test_describe_tuning(cents, text):
    assert rhythm.describe_tuning(cents) == text


def test_wrap_cents():
    assert rhythm.wrap_cents(60) == -40 and rhythm.wrap_cents(-55) == 45


# ------------------------------------------------------------------ segments
def test_frames_to_segments():
    segs = frames_to_segments(["C", "C", "G", "G", "G", "N"], 0.5, duration=2.8)
    assert segs == [(0.0, 1.0, "C"), (1.0, 2.5, "G"), (2.5, 2.8, "N")]


def test_merge_neighbours_joins_equal_labels():
    assert merge_neighbours([(0, 1, "C"), (1, 2, "C"), (2, 2, "G"), (2, 3, "F")]) == [(0, 2, "C"), (2, 3, "F")]


def test_remove_short_flicker_between_same_chord():
    segs = [(0, 2, "C"), (2, 2.1, "A:min"), (2.1, 4, "C"), (4, 6, "G")]
    assert remove_short(segs, 0.4) == [(0, 4, "C"), (4, 6, "G")]


def test_remove_short_goes_to_longer_neighbour():
    segs = [(0, 1, "C"), (1, 1.2, "E:min"), (1.2, 4, "G")]
    assert remove_short(segs, 0.4) == [(0, 1, "C"), (1, 4, "G")]


def test_remove_short_keeps_everything_long_enough():
    segs = [(0, 1, "C"), (1, 2, "G")]
    assert remove_short(segs, 0.5) == segs and remove_short(segs, 0) == segs


def test_snap_to_beats():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    segs = [(0.0, 0.93, "C"), (0.93, 2.07, "G"), (2.07, 3.0, "F")]
    assert snap_to_beats(segs, beats) == [(0.0, 1.0, "C"), (1.0, 2.0, "G"), (2.0, 3.0, "F")]


def test_snap_leaves_distant_changes_alone():
    beats = [0.0, 1.0, 2.0, 3.0]
    segs = [(0.0, 1.5, "C"), (1.5, 3.0, "G")]
    assert snap_to_beats(segs, beats) == segs


def test_segment_at():
    segs = [(0.0, 1.0, "C"), (1.0, 2.0, "G"), (2.0, 3.0, "F")]
    starts = [s for s, _, _ in segs]
    assert segment_at(segs, starts, -0.1) == -1
    assert segment_at(segs, starts, 0.0) == 0
    assert segment_at(segs, starts, 1.5) == 1
    assert segment_at(segs, starts, 2.99) == 2
