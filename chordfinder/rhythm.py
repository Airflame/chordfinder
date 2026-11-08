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


def meter_from_downbeats(beats, downbeats) -> int | None:
    """Most common number of beats per bar, or None if unknown."""
    b = np.asarray(beats, dtype=float)
    d = np.asarray(downbeats, dtype=float)
    if b.size < 4 or d.size < 3:
        return None
    counts = []
    for start, end in zip(d[:-1], d[1:]):
        n = int(np.sum((b >= start - 0.03) & (b < end - 0.03)))
        if 2 <= n <= 12:
            counts.append(n)
    if not counts:
        return None
    values, freq = np.unique(counts, return_counts=True)
    return int(values[np.argmax(freq)])


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
