"""Turning frame-wise chord predictions into a clean chord timeline.

A segment is a tuple (start, end, label) with times in seconds and a
Harte-style chord label ("C:maj7", "N" for no chord).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Segment = tuple[float, float, str]


def frames_to_segments(labels: Sequence[str], frame_seconds: float, duration: float | None = None) -> list[Segment]:
    """Merge runs of identical frame labels into segments."""
    segments: list[Segment] = []
    if not len(labels):
        return segments
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segments.append((start * frame_seconds, i * frame_seconds, labels[start]))
            start = i
    if duration is not None:
        segments = [(s, min(e, duration), lab) for s, e, lab in segments if s < duration]
    return segments


def merge_neighbours(segments: list[Segment]) -> list[Segment]:
    """Join adjacent segments that carry the same label and drop empty ones."""
    out: list[Segment] = []
    for s, e, lab in segments:
        if e - s <= 1e-6:
            continue
        if out and out[-1][2] == lab and abs(out[-1][1] - s) < 1e-3:
            out[-1] = (out[-1][0], e, lab)
        else:
            out.append((s, e, lab))
    return out


def remove_short(segments: list[Segment], min_seconds: float) -> list[Segment]:
    """Absorb segments shorter than min_seconds into a neighbour.

    The shortest segment goes first, into the neighbour it shares a label with
    if any, otherwise into the longer neighbour, so flickers between two chords
    disappear without shifting real chord changes.
    """
    segs = merge_neighbours(list(segments))
    if min_seconds <= 0:
        return segs
    while len(segs) > 1:
        lengths = [e - s for s, e, _ in segs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_seconds:
            break
        s, e, lab = segs[i]
        prev = segs[i - 1] if i > 0 else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if prev and nxt and prev[2] == nxt[2]:
            segs[i - 1:i + 2] = [(prev[0], nxt[1], prev[2])]
        elif prev and (not nxt or prev[1] - prev[0] >= nxt[1] - nxt[0]):
            segs[i - 1:i + 1] = [(prev[0], e, prev[2])]
        else:
            segs[i:i + 2] = [(s, nxt[1], nxt[2])]
        segs = merge_neighbours(segs)
    return segs


def snap_to_beats(segments: list[Segment], beats: Sequence[float], tolerance: float = 0.35) -> list[Segment]:
    """Move each chord change onto the nearest beat.

    tolerance is a fraction of the local beat period; changes further from any
    beat stay where they are.
    """
    b = np.asarray(beats, dtype=float)
    if b.size < 2 or len(segments) < 2:
        return list(segments)
    period = float(np.median(np.diff(b)))
    out = [list(seg) for seg in segments]
    for k in range(1, len(out)):
        t = out[k][0]
        j = int(np.argmin(np.abs(b - t)))
        if abs(b[j] - t) <= tolerance * period:
            t = float(b[j])
        # Never cross the neighbouring changes.
        t = min(max(t, out[k - 1][0]), out[k][1])
        out[k - 1][1] = t
        out[k][0] = t
    return merge_neighbours([tuple(s) for s in out])


def segment_at(segments: Sequence[Segment], starts: Sequence[float], t: float) -> int:
    """Index of the segment playing at time t (-1 before the first one)."""
    i = int(np.searchsorted(starts, t, side="right")) - 1
    if i < 0 or i >= len(segments):
        return -1
    return i if t < segments[i][1] or i == len(segments) - 1 else -1
