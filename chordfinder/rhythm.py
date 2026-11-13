"""Tempo and meter from beat and downbeat times, and tuning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STABLE_TOLERANCE = 0.04  # ±4 % tempo drift still counts as a steady tempo


@dataclass
class Tempo:
    bpm: float          # 0 when unknown
    low: float          # 10th percentile of the local tempo
    high: float         # 90th percentile
    stable: bool

    def text(self) -> str:
        if self.bpm <= 0:
            return "–"
        bpm = format_bpm(self.bpm)
        if self.stable:
            return f"{bpm} BPM"
        return f"≈{bpm} BPM ({self.low:.0f}–{self.high:.0f})"


def format_bpm(bpm: float) -> str:
    """128 BPM stays "128"; 127.6 is shown with one decimal."""
    return f"{round(bpm)}" if abs(bpm - round(bpm)) < 0.15 else f"{bpm:.1f}"


def tempo_from_beats(beats) -> Tempo:
    """Tempo of a beat sequence (times in seconds).

    The tempo is the song's main beat level. Passages that a beat tracker taps
    in double time (extra beats) or half time (missed beats) are folded back
    onto that level, so they don't count as tempo changes. A steady tempo is
    then measured with a least-squares fit over all beats, which is far more
    precise than averaging single intervals.
    """
    b = np.sort(np.asarray(beats, dtype=float))
    if b.size < 4:
        return Tempo(0.0, 0.0, 0.0, False)
    median = float(np.median(np.diff(b)))
    if median <= 0:
        return Tempo(0.0, 0.0, 0.0, False)
    kept = [b[0]]
    for t in b[1:]:
        if t - kept[-1] >= 0.75 * median:   # drop extra beats in double-time passages
            kept.append(t)
    b = np.asarray(kept)
    if b.size < 4:
        return Tempo(60.0 / median, 60.0 / median, 60.0 / median, False)
    ibi = np.diff(b)
    steps = np.maximum(1, np.round(ibi / median))  # missed beats: one interval spans several beats
    period = ibi / steps
    # Local tempo over 8-beat windows smooths out single-beat jitter.
    window = min(8, max(1, period.size // 2))
    local = [60.0 / float(np.median(period[i:i + window])) for i in range(0, period.size - window + 1)]
    low, high = float(np.percentile(local, 10)), float(np.percentile(local, 90))
    bpm_median = 60.0 / float(np.median(period))
    stable = (high - low) / bpm_median <= 2 * STABLE_TOLERANCE
    bpm = bpm_median
    if stable:
        index = np.concatenate([[0], np.cumsum(steps)])
        slope = float(np.polyfit(index, b, 1)[0])
        if slope > 0 and abs(60.0 / slope - bpm_median) < STABLE_TOLERANCE * bpm_median:
            bpm = 60.0 / slope
    return Tempo(bpm, low, high, stable)


MAX_METER = 12            # longest bar that counts as a meter, in beats (default of the "Longest bar" option)
REGULAR_SHARE = 0.8       # this share of equally long bars means the downbeats need no repair
MAX_BAR = 16              # upper limit of that option
METER_CHANGE_PENALTY = 1.0  # in units of one certain downbeat, see regroup_downbeats


def bar_lengths(beats, downbeats) -> list[int]:
    """Number of beats in each bar (the bar after the last downbeat is left out)."""
    b = np.asarray(beats, dtype=float)
    d = np.asarray(downbeats, dtype=float)
    if b.size == 0 or d.size < 2:
        return []
    first = np.searchsorted(b, d - 0.03)
    return [int(n) for n in np.diff(first)]


def meters(beats, downbeats, max_meter: int = MAX_METER) -> list[int]:
    """Beats per bar, most common first. Only meters that make up a real part of the song
    are listed, so a few miscounted bars don't show up as a meter change."""
    counts = [n for n in bar_lengths(beats, downbeats) if 2 <= n <= max_meter]
    if len(beats) < 4 or len(counts) < 2:
        return []
    values, freq = np.unique(counts, return_counts=True)
    order = np.argsort(-freq, kind="stable")
    main = int(values[order[0]])
    others = [int(values[i]) for i in order[1:] if freq[i] >= max(4, 0.2 * len(counts))]
    return [main] + others


def meter_from_downbeats(beats, downbeats, max_meter: int = MAX_METER) -> int | None:
    """Most common number of beats per bar, or None if unknown."""
    found = meters(beats, downbeats, max_meter)
    return found[0] if found else None


def meter_text(found: list[int]) -> str:
    """"4/4", "7/4", or "7/4 and 4/4" for a song that changes meter."""
    return " and ".join(f"{n}/4" for n in found[:3]) if found else "–"


def regroup_downbeats(beats, downbeats, strength=None, max_meter: int = MAX_METER) -> np.ndarray:
    """Repair the downbeats of a song in an uncommon meter.

    Beat trackers know 4/4 and 3/4 best. In 5/4 or 7/4 they tend to keep the
    beats right but also mark accents inside the bar as downbeats, so a 5/4
    bar comes out as a 3-beat and a 2-beat bar. When the bars are irregular,
    the downbeats are chosen again: the chain of beats that carries the most
    downbeat evidence while changing its bar length as rarely as possible.

    strength: how sure the tracker is that each beat is a downbeat (0…1, one
    value per beat). Without it the detected downbeats count as 1.
    max_meter: the longest bar to build, in beats. Bars the tracker made longer
    than this are split as well.
    Songs whose bars are already regular (and short enough) are returned unchanged.
    """
    max_meter = int(min(max(max_meter, 2), MAX_BAR))
    b = np.asarray(beats, dtype=float)
    d = np.asarray(downbeats, dtype=float)
    lengths = bar_lengths(b, d)
    if b.size < 12 or len(lengths) < 4:
        return d
    values, freq = np.unique(lengths, return_counts=True)
    too_long = int(values[np.argmax(freq)]) > max_meter
    if freq.max() / len(lengths) >= REGULAR_SHARE and not too_long:
        return d

    if strength is not None and len(strength) == b.size:
        weight = np.clip(np.asarray(strength, dtype=float), 0.0, 1.0)
    else:
        weight = np.zeros(b.size)
        weight[np.clip(np.searchsorted(b, d - 0.03), 0, b.size - 1)] = 1.0
    # Only clearly stronger beats count for a bar line, ordinary beats count against it.
    # Otherwise short bars would win just by having more downbeats.
    gain = weight - (weight.mean() + 0.5 * weight.std())

    # score[i, n]: best chain that ends with a downbeat on beat i after a bar of n beats.
    # Keeping the bar length is free; every change of length costs METER_CHANGE_PENALTY.
    n_beats = b.size
    score = np.full((n_beats, max_meter + 1), -np.inf)
    previous = np.full((n_beats, max_meter + 1), -1)    # bar length before, 0 = the chain starts there
    best_any = np.full(n_beats, -np.inf)
    for i in range(n_beats):
        for n in range(2, min(max_meter, i) + 1):
            j = i - n
            options = (gain[j], score[j, n], best_any[j] - METER_CHANGE_PENALTY)
            k = int(np.argmax(options))
            score[i, n] = options[k] + gain[i]
            previous[i, n] = (0, n, int(np.argmax(score[j])))[k]
        best_any[i] = score[i].max()
    i = int(np.argmax(best_any))
    if not np.isfinite(best_any[i]):
        return d
    n = int(np.argmax(score[i]))
    chain = [i]
    while n:
        i, n = i - n, int(previous[i, n])
        chain.append(i)
    chain = chain[::-1]
    # Quiet passages at either end score nothing: carry the bars on as far as the tracker had downbeats.
    if len(chain) > 1:
        first, last = np.searchsorted(b, d[[0, -1]] - 0.03)
        step = chain[1] - chain[0]
        while chain[0] - step >= first:
            chain.insert(0, chain[0] - step)
        step = chain[-1] - chain[-2]
        while chain[-1] + step <= min(last, n_beats - 1):
            chain.append(chain[-1] + step)
    regrouped = b[chain]

    # Keep the result only if the bars became steadier (fewer changes of length from bar to bar),
    # or if they had to be shortened.
    def unsteadiness(bars: list[int]) -> float:
        return float(np.count_nonzero(np.diff(bars))) / max(1, len(bars) - 1)

    after = bar_lengths(b, regrouped)
    if len(after) < 4 or (unsteadiness(after) >= unsteadiness(lengths) and not too_long):
        return d
    return regrouped


def bar_position(t: float, beats, downbeats) -> tuple[int, int] | None:
    """(bar number, beat in bar), both 1-based, for a time in seconds."""
    if len(beats) == 0 or t < beats[0] - 0.05:
        return None
    beat_idx = int(np.searchsorted(beats, t + 0.02, side="right")) - 1
    if len(downbeats) == 0:
        return None
    bar_idx = int(np.searchsorted(downbeats, beats[beat_idx] + 0.02, side="right")) - 1
    if bar_idx < 0:
        return 0, beat_idx + 1  # pickup beats before the first downbeat
    first_beat_of_bar = int(np.searchsorted(beats, downbeats[bar_idx] - 0.02))
    return bar_idx + 1, beat_idx - first_beat_of_bar + 1


# --------------------------------------------------------------- tuning
REFERENCE_A4 = 440.0


def cents_to_a4(cents: float) -> float:
    return REFERENCE_A4 * 2 ** (cents / 1200)


def describe_tuning(cents: float) -> str:
    """e.g. "A = 440 Hz (standard)" or "A = 432.1 Hz (−32 ¢)"."""
    hz = cents_to_a4(cents)
    if abs(cents) < 4:
        return f"A = {hz:.1f} Hz (standard)"
    sign = "+" if cents > 0 else "−"
    return f"A = {hz:.1f} Hz ({sign}{abs(cents):.0f} ¢)"


def short_tuning(cents: float) -> str:
    """Compact form for tables: "440 Hz", "432.1 Hz"."""
    hz = cents_to_a4(cents)
    return f"{hz:.0f} Hz" if abs(cents) < 4 else f"{hz:.1f} Hz"


def wrap_cents(cents: float) -> float:
    """Fold a deviation into [-50, 50) cents."""
    return (cents + 50.0) % 100.0 - 50.0


def semitone_ratio(cents: float) -> float:
    return math.pow(2.0, cents / 1200.0)
