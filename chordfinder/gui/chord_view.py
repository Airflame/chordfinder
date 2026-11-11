"""Real-time chord display widgets for the player."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import rhythm, theory
from ..theory import Chord, Key

# Function families and their colours.
TONIC, SUBDOMINANT, DOMINANT, SECONDARY, CHROMATIC, NONE = range(6)
FAMILY_COLORS = {
    TONIC: QColor("#27AE60"),
    SUBDOMINANT: QColor("#F1C40F"),
    DOMINANT: QColor("#E74C3C"),
    SECONDARY: QColor("#D35400"),
    CHROMATIC: QColor("#2980B9"),
    NONE: QColor("#8A9099"),
}
FAMILY_NAMES = {TONIC: "tonic", SUBDOMINANT: "subdominant", DOMINANT: "dominant",
                SECONDARY: "secondary dominant", CHROMATIC: "borrowed / chromatic", NONE: "no chord"}
_DEGREE_FAMILY = {"I": TONIC, "III": TONIC, "VI": TONIC, "II": SUBDOMINANT, "IV": SUBDOMINANT,
                  "V": DOMINANT, "VII": DOMINANT}


@dataclass
class ChordItem:
    start: float
    end: float
    symbol: str      # "Fmaj7"
    numeral: str     # "IVmaj7" (empty without a key or for N.C.)
    family: int


def family_of(numeral: str) -> int:
    if not numeral:
        return NONE
    if "/" in numeral:
        return SECONDARY
    if numeral[0] in "b#":
        return CHROMATIC
    letters = ""
    for ch in numeral:
        if ch.upper() in "IV":
            letters += ch.upper()
        else:
            break
    return _DEGREE_FAMILY.get(letters, CHROMATIC)


def build_items(chords: list[tuple[float, float, str]], key: Key | None, simplify: bool = False) -> list[ChordItem]:
    items = []
    for start, end, label in chords:
        chord = Chord.parse(label)
        if simplify:
            chord = chord.simplified()
        numeral = theory.function(chord, key) if key and chord.is_chord else ""
        items.append(ChordItem(start, end, chord.symbol(key), numeral, family_of(numeral)))
    # Simplifying can make neighbours identical; show them as one chord.
    merged: list[ChordItem] = []
    for it in items:
        if merged and merged[-1].symbol == it.symbol and merged[-1].numeral == it.numeral:
            merged[-1].end = it.end
        else:
            merged.append(it)
    return merged


def _mix(a: QColor, b: QColor, f: float) -> QColor:
    return QColor(round(a.red() * (1 - f) + b.red() * f), round(a.green() * (1 - f) + b.green() * f),
                  round(a.blue() * (1 - f) + b.blue() * f))


class _Base(QWidget):
    """Shared state: chord items, beats and the current time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[ChordItem] = []
        self.starts: list[float] = []
        self.beats: list[float] = []
        self.downbeats: list[float] = []
        self._beats_arr = np.zeros(0)
        self._downbeats_arr = np.zeros(0)
        self._down_set: set[float] = set()
        self.meter = 4
        self.time = 0.0
        self.show_functions = True
        self.message = "Analyse a song and play it to see its chords here."

    def set_song(self, items: list[ChordItem], beats=(), downbeats=(), meter: int | None = None,
                 message: str = "") -> None:
        self.items = items
        self.starts = [it.start for it in items]
        self.beats = [float(b) for b in beats]
        self.downbeats = [float(d) for d in downbeats]
        # Cached for the per-frame paint code.
        self._beats_arr = np.asarray(self.beats, dtype=float)
        self._downbeats_arr = np.asarray(self.downbeats, dtype=float)
        self._down_set = {round(d, 3) for d in self.downbeats}
        self.meter = meter or 4
        self.message = message
        self.update()

    def set_time(self, t: float) -> None:
        self.time = t
        self.update()

    def index_at(self, t: float) -> int:
        i = bisect.bisect_right(self.starts, t) - 1
        if i < 0 or i >= len(self.items):
            return -1
        return i if t < self.items[i].end or i == len(self.items) - 1 else -1

    def _palette(self) -> tuple[QColor, QColor, QColor, bool]:
        pal = self.palette()
        bg = pal.color(QPalette.Base)
        fg = pal.color(QPalette.Text)
        mid = _mix(bg, fg, 0.45)
        return bg, fg, mid, bg.lightness() < 128


class ChordDisplay(_Base):
    """The chord playing now (large), its function, the next chord and the beat in the bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(132)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg, fg, mid, dark = self._palette()
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 10, 10)
        p.fillPath(path, bg)
        p.setPen(QPen(_mix(bg, fg, 0.18), 1))
        p.drawPath(path)

        if not self.items:
            p.setPen(mid)
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return

        i = self.index_at(self.time)
        current = self.items[i] if i >= 0 else None
        if i < 0:  # between chords: the next one is the first that starts later
            j = bisect.bisect_right(self.starts, self.time)
        else:
            j = i + 1
        last = len(self.items) - 1
        while j < last and self.items[j].family == NONE:
            j += 1  # skip silences when showing what comes next
        nxt = self.items[j] if j <= last else None

        w, h = r.width(), r.height()
        split = w * 0.62
        accent = FAMILY_COLORS[current.family] if current else mid

        # Accent bar on the left in the colour of the function family.
        p.fillRect(QRectF(r.left() + 10, r.top() + 14, 5, h - 28), accent)

        # Current chord
        big = QFont(self.font())
        big.setPointSizeF(max(20.0, min(40.0, h * 0.30)))
        big.setBold(True)
        p.setFont(big)
        p.setPen(fg)
        symbol = current.symbol if current else "–"
        top = QRectF(r.left() + 28, r.top() + 6, split - 36, h * 0.52)
        p.drawText(top, Qt.AlignLeft | Qt.AlignBottom, symbol)

        sub = QFont(self.font())
        sub.setPointSizeF(max(11.0, min(18.0, h * 0.14)))
        p.setFont(sub)
        if current and current.numeral and self.show_functions:
            p.setPen(accent if not dark else accent.lighter(125))
            func = current.numeral
            p.drawText(QRectF(r.left() + 28, top.bottom() + 2, split - 36, h * 0.25),
                       Qt.AlignLeft | Qt.AlignTop, f"{func}  ·  {FAMILY_NAMES[current.family]}")

        # Beat dots and bar number
        pos = rhythm.bar_position(self.time, self._beats_arr, self._downbeats_arr) if self.downbeats else None
        dots_y = r.bottom() - 16
        if pos:
            bar, beat = pos
            small = QFont(self.font())
            small.setPointSizeF(8.5)
            p.setFont(small)
            p.setPen(mid)
            label = f"Bar {bar}" if bar > 0 else "Pickup"
            p.drawText(QRectF(r.left() + 28, dots_y - 8, 60, 16), Qt.AlignLeft | Qt.AlignVCenter, label)
            for k in range(self.meter):
                on = k + 1 == beat
                color = accent if on else _mix(bg, fg, 0.2)
                p.setBrush(color)
                p.setPen(Qt.NoPen)
                radius = 5.5 if k == 0 else 4.5
                p.drawEllipse(QPointF(r.left() + 96 + k * 18, dots_y), radius, radius)

        # Divider and the next chord
        p.setPen(QPen(_mix(bg, fg, 0.15), 1))
        p.drawLine(QPointF(r.left() + split, r.top() + 14), QPointF(r.left() + split, r.bottom() - 14))
        right = QRectF(r.left() + split + 16, r.top() + 8, w - split - 26, h - 16)
        small = QFont(self.font())
        small.setPointSizeF(8.5)
        p.setFont(small)
        p.setPen(mid)
        p.drawText(QRectF(right.left(), right.top() + 4, right.width(), 16), Qt.AlignLeft, "NEXT")
        if nxt is not None:
            mid_font = QFont(self.font())
            mid_font.setPointSizeF(max(14.0, min(22.0, h * 0.18)))
            mid_font.setBold(True)
            p.setFont(mid_font)
            p.setPen(fg)
            p.drawText(QRectF(right.left(), right.top() + 20, right.width(), h * 0.3),
                       Qt.AlignLeft | Qt.AlignVCenter, nxt.symbol)
            p.setFont(sub)
            line = []
            if nxt.numeral and self.show_functions:
                line.append(nxt.numeral)
            wait = nxt.start - self.time
            if 0 < wait < 60:
                line.append(f"in {wait:.1f} s")
            p.setPen(mid)
            p.drawText(QRectF(right.left(), right.top() + 20 + h * 0.3, right.width(), h * 0.25),
                       Qt.AlignLeft | Qt.AlignTop, "   ".join(line))


class ChordTimeline(_Base):
    """A scrolling strip of chords around the playhead. Click to seek, wheel to zoom."""

    seek_requested = Signal(float)
    PLAYHEAD = 0.3  # playhead position as a fraction of the width

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seconds_visible = 12.0
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to jump to a chord. Scroll the mouse wheel to zoom.")

    def _x(self, t: float) -> float:
        w = self.width()
        return w * self.PLAYHEAD + (t - self.time) * w / self.seconds_visible

    def _t(self, x: float) -> float:
        w = self.width()
        return self.time + (x - w * self.PLAYHEAD) * self.seconds_visible / w

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.items:
            t = self._t(event.position().x())
            i = self.index_at(t)
            # Clicking a chord jumps to its start, clicking empty space to that time.
            self.seek_requested.emit(max(0.0, self.items[i].start if i >= 0 else t))

    def wheelEvent(self, event) -> None:
        factor = 0.85 if event.angleDelta().y() > 0 else 1 / 0.85
        self.seconds_visible = max(4.0, min(60.0, self.seconds_visible * factor))
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg, fg, mid, dark = self._palette()
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        frame = QPainterPath()
        frame.addRoundedRect(r, 10, 10)
        p.fillPath(frame, bg)
        p.setClipPath(frame)

        if not self.items:
            p.setPen(mid)
            p.drawText(r, Qt.AlignCenter, "Chord timeline")
            p.setClipping(False)
            p.setPen(QPen(_mix(bg, fg, 0.18), 1))
            p.drawPath(frame)
            return

        t0, t1 = self._t(0), self._t(self.width())
        grid_top, grid_h = r.top() + 18, 10
        block_top = grid_top + grid_h + 6
        block_h = r.bottom() - block_top - 8

        # Beat grid with bar numbers
        small = QFont(self.font())
        small.setPointSizeF(7.5)
        p.setFont(small)
        down = self._down_set
        lo = bisect.bisect_left(self.beats, t0)
        hi = bisect.bisect_right(self.beats, t1)
        for b in self.beats[lo:hi]:
            x = self._x(b)
            is_down = round(b, 3) in down
            p.setPen(QPen(_mix(bg, fg, 0.55 if is_down else 0.22), 1.4 if is_down else 1))
            p.drawLine(QPointF(x, grid_top + (0 if is_down else 4)), QPointF(x, grid_top + grid_h))
            if is_down:
                bar = bisect.bisect_right(self.downbeats, b + 0.001)
                p.setPen(mid)
                p.drawText(QRectF(x + 3, r.top() + 2, 40, 14), Qt.AlignLeft | Qt.AlignVCenter, str(bar))

        # Chord blocks
        first = max(0, bisect.bisect_right(self.starts, t0) - 1)
        name_font = QFont(self.font())
        name_font.setPointSizeF(10.5)
        name_font.setBold(True)
        num_font = QFont(self.font())
        num_font.setPointSizeF(8.5)
        current = self.index_at(self.time)
        for i in range(first, len(self.items)):
            it = self.items[i]
            if it.start > t1:
                break
            x0, x1 = self._x(it.start) + 1.5, self._x(it.end) - 1.5
            if x1 <= x0:
                continue
            base = FAMILY_COLORS[it.family]
            active = i == current
            fill = _mix(bg, base, 0.85 if active else (0.45 if dark else 0.28))
            block = QPainterPath()
            block.addRoundedRect(QRectF(x0, block_top, x1 - x0, block_h), 6, 6)
            p.fillPath(block, fill)
            if active:
                p.setPen(QPen(base.darker(115), 1.5))
                p.drawPath(block)
            text_color = QColor("white") if active else fg
            inner = QRectF(x0 + 6, block_top + 2, max(0.0, x1 - x0 - 10), block_h - 4)
            if inner.width() > 14:
                p.setPen(text_color)
                p.setFont(name_font)
                show_num = self.show_functions and it.numeral and block_h > 34
                name_rect = QRectF(inner.left(), inner.top(), inner.width(), inner.height() * (0.55 if show_num else 1))
                p.drawText(name_rect, Qt.AlignLeft | (Qt.AlignBottom if show_num else Qt.AlignVCenter),
                           p.fontMetrics().elidedText(it.symbol if it.family != NONE else "N.C.",
                                                      Qt.ElideRight, int(inner.width())))
                if show_num:
                    p.setFont(num_font)
                    p.drawText(QRectF(inner.left(), name_rect.bottom() + 1, inner.width(), inner.height() * 0.45),
                               Qt.AlignLeft | Qt.AlignTop,
                               p.fontMetrics().elidedText(it.numeral, Qt.ElideRight, int(inner.width())))

        # Playhead
        x = self._x(self.time)
        p.setPen(QPen(QColor("#E5484D"), 2))
        p.drawLine(QPointF(x, r.top() + 2), QPointF(x, r.bottom() - 2))
        p.setClipping(False)
        p.setPen(QPen(_mix(bg, fg, 0.18), 1))
        p.drawPath(frame)
