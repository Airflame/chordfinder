"""Main application window."""

from __future__ import annotations

import bisect
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDragEnterEvent, QDropEvent, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QStyle, QStyleOptionSlider, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .chord_view import (
    CHROMATIC, DEFAULT_FAMILY_COLORS, DOMINANT, FAMILY_COLORS, FAMILY_NAMES, SECONDARY, SUBDOMINANT, TONIC,
    ChordDisplay, ChordTimeline, build_items,
)
from .worker import Worker
from .. import audio_io, model_store, models, rhythm
from ..analyzer import Analysis, Pipeline, Settings, chart_path, default_cache_dir, default_models_dir, load_cached
from ..theory import ALL_KEYS, Key

COL_FILE, COL_STATUS, COL_BPM, COL_KEY, COL_CAMELOT, COL_TUNING = range(6)
PATH_ROLE = Qt.UserRole
RESULT_ROLE = Qt.UserRole + 1
KEY_ROLE = Qt.UserRole + 2        # the user's key override for this song (Key.index or None)
ICON_PATH = Path(__file__).resolve().parent.parent / "resources" / "icon.png"
# Chord-function families whose colours the user can change (shown as the legend).
LEGEND_FAMILIES = (TONIC, SUBDOMINANT, DOMINANT, SECONDARY, CHROMATIC)


class FunctionThread(QThread):
    """Runs a callable off the UI thread and reports its result or error."""
    finished_ok = Signal(object)
    failed = Signal(str)
    message = Signal(str)
    progress = Signal(float)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            self.finished_ok.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e) or type(e).__name__)


class SeekSlider(QSlider):
    """A slider that jumps to where the groove is clicked (instead of paging) and can be dragged on from there."""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.maximum() > self.minimum():
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            if not handle.contains(event.position().toPoint()):
                groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
                x = event.position().x() - groove.left() - handle.width() / 2
                span = groove.width() - handle.width()
                value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(x), max(1, span),
                                                       opt.upsideDown)
                self.setSliderPosition(value)  # triggers SliderMove -> actionTriggered -> seek
        # The handle is now under the cursor, so the default handling starts a drag from here.
        super().mousePressEvent(event)


def _probe_gpu() -> str | None:
    import torch
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} ({props.total_memory / 2**30:.1f} GB)"


def _fmt_time(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def _fmt_seconds(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


class MainWindow(QMainWindow):
    def __init__(self, files: list[str] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Chordfinder")
        self.resize(1320, 860)
        self.setAcceptDrops(True)
        self.qsettings = QSettings("Chordfinder", "Chordfinder")
        self.pipeline = Pipeline()
        self.worker: Worker | None = None
        self.helper: FunctionThread | None = None
        self.gpu_thread: FunctionThread | None = None
        self.player = None  # created lazily; QtMultimedia loads FFmpeg on first use
        self.current_row: int | None = None   # the song shown in the player and the Chords tab
        self._chord_row = -1                  # highlighted row in the chord list

        # ~60 fps; a precise timer because Windows' coarse timers snap to ~15.6 ms slots and stutter.
        self.tick = QTimer(self)
        self.tick.setTimerType(Qt.PreciseTimer)
        self.tick.setInterval(16)
        self.tick.timeout.connect(self._on_tick)
        # QMediaPlayer.position() advances in coarse steps, so the display runs its own clock
        # between them (see _display_time).
        self._clock = QElapsedTimer()
        self._clock_pos = 0.0         # displayed time when _clock was last restarted
        self._last_reported = -1.0    # last raw player position seen

        self._build_ui()
        self._load_settings()
        self._update_model_widgets()
        self._update_buttons()
        self._show_song(None)
        self._probe_gpu()
        if files:
            self._add_paths([Path(f) for f in files])

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        # Songs
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Song", "Status", "BPM", "Key", "Camelot", "Tuning"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        for col in (COL_STATUS, COL_BPM, COL_KEY, COL_CAMELOT, COL_TUNING):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(lambda idx: self._play_row(idx.row()))
        self.table.itemSelectionChanged.connect(self._on_selection)
        self.table.setToolTip("Double-click a song to play it with its chords.")

        self.add_btn = QPushButton("Add songs…")
        self.add_btn.clicked.connect(self._pick_files)
        self.add_folder_btn = QPushButton("Add folder…")
        self.add_folder_btn.clicked.connect(self._pick_folder)
        self.remove_btn = QPushButton("Remove")
        self.clear_btn = QPushButton("Clear")
        self.export_btn = QPushButton("Export chords…")
        self.export_btn.setToolTip("Save the chords of the selected song as a chord chart, .lab or .json file.")
        self.folder_btn = QPushButton("Show in folder")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn.clicked.connect(self._clear)
        self.export_btn.clicked.connect(self._export)
        self.folder_btn.clicked.connect(self._show_in_folder)
        song_buttons = QHBoxLayout()
        for b in (self.add_btn, self.add_folder_btn, self.remove_btn, self.clear_btn):
            song_buttons.addWidget(b)
        song_buttons.addStretch()
        song_buttons.addWidget(self.export_btn)
        song_buttons.addWidget(self.folder_btn)

        songs_box = QGroupBox("Songs")
        sv = QVBoxLayout(songs_box)
        sv.addWidget(self.table)
        sv.addLayout(song_buttons)

        # Player with the real-time chord display
        self.chord_display = ChordDisplay()
        self.timeline = ChordTimeline()
        self.timeline.seek_requested.connect(self._seek_seconds)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setMinimumWidth(90)
        self.play_btn.clicked.connect(self._toggle_play)
        self.seek = SeekSlider(Qt.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.setSingleStep(5000)   # arrow keys / wheel: 5 s
        self.seek.setPageStep(10000)    # Page Up/Down: 10 s
        # Any user action (click, drag, keys, wheel) seeks; sliderPosition already holds the new value here.
        self.seek.actionTriggered.connect(lambda _action: self._seek_ms(self.seek.sliderPosition()))
        self.time_label = QLabel("0:00 / 0:00")
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(90)
        self.volume.setToolTip("Volume")
        self.volume.valueChanged.connect(self._set_volume)
        player_row = QHBoxLayout()
        player_row.addWidget(self.play_btn)
        player_row.addWidget(self.seek, 1)
        player_row.addWidget(self.time_label)
        player_row.addWidget(QLabel("Volume"))
        player_row.addWidget(self.volume)
        self.now_playing = QLabel("")
        self.now_playing.setMinimumWidth(10)
        self.summary_label = QLabel("")
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        info_row = QHBoxLayout()
        info_row.addWidget(self.now_playing, 1)
        info_row.addWidget(self.summary_label)

        player_box = QGroupBox("Player")
        pv = QVBoxLayout(player_box)
        pv.addWidget(self.chord_display)
        pv.addWidget(self.timeline)
        pv.addLayout(player_row)
        pv.addLayout(info_row)

        # Options
        self.tabs = QTabWidget()
        self.tabs.addTab(self._chords_tab(), "Chords")
        self.tabs.addTab(self._analysis_tab(), "Analysis")
        self.tabs.addTab(self._models_tab(), "Models && performance")

        # Progress + log
        self.current_label = QLabel("Idle")
        self.job_bar = QProgressBar()
        self.total_bar = QProgressBar()
        self.total_bar.setFormat("Overall: %v / %m songs")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setMinimumHeight(60)

        self.start_btn = QPushButton("Analyze songs")
        self.start_btn.setDefault(True)
        self.start_btn.setMinimumHeight(34)
        self.start_btn.setToolTip("Analyze every song that hasn't been analyzed yet. If all are done, "
                                  "the selected songs are analyzed again.")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn.clicked.connect(self._cancel)
        run_row = QHBoxLayout()
        run_row.addWidget(self.current_label, 1)
        run_row.addWidget(self.cancel_btn)
        run_row.addWidget(self.start_btn)

        # Left: songs, player, progress, log. Right: options.
        left = QVBoxLayout()
        left.addWidget(songs_box, 3)
        left.addWidget(player_box)
        left.addLayout(run_row)
        left.addWidget(self.job_bar)
        left.addWidget(self.total_bar)
        left.addWidget(self.log_view, 1)

        right = QVBoxLayout()
        right.addWidget(self.tabs)
        self.tabs.setFixedWidth(480)

        columns = QHBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right)
        root.addLayout(columns)

        for text, keys, slot in (
                ("Quit", "Ctrl+Q", self.close),
                ("Play/pause", "Ctrl+Space", self._toggle_play),
                ("Back 5 s", "Ctrl+Left", lambda: self._skip(-5)),
                ("Forward 5 s", "Ctrl+Right", lambda: self._skip(5)),
                ("Add songs", "Ctrl+O", self._pick_files)):
            self.addAction(QAction(text, self, shortcut=QKeySequence(keys), triggered=slot))
        play_key = QAction("Play/pause", self, shortcut=QKeySequence(Qt.Key_MediaTogglePlayPause),
                           triggered=self._toggle_play)
        self.addAction(play_key)

    def _chords_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        form = QFormLayout()
        self.key_value = QLabel("–")
        self.key_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.key_alt_value = QLabel("")
        self.key_alt_value.setWordWrap(True)
        self.tempo_value = QLabel("–")
        self.tuning_value = QLabel("–")
        self.tuning_value.setToolTip("The reference pitch the song is tuned to. Recordings tuned more than half "
                                     "a semitone away (for example a semitone down) show up as a different key.")
        for label in (self.key_value, self.tempo_value, self.tuning_value):
            font = label.font()
            font.setPointSizeF(font.pointSizeF() + 1.5)
            font.setBold(True)
            label.setFont(font)
        self.function_key_combo = QComboBox()
        self.function_key_combo.setToolTip("The key the Roman numerals refer to. Choose another key if the song "
                                           "modulates or the detected key is the relative major/minor.")
        self.function_key_combo.addItem("Detected key", None)
        for k in sorted(ALL_KEYS, key=lambda k: (int(k.camelot[:-1]), k.camelot[-1])):
            self.function_key_combo.addItem(f"{k.name}  ({k.camelot})", k.index)
        self.function_key_combo.activated.connect(self._on_function_key)
        self.show_functions_chk = QCheckBox("Show chord functions (Roman numerals)")
        self.simplify_chk = QCheckBox("Simplify to triads (Fmaj7 → F)")
        self.show_functions_chk.toggled.connect(self._refresh_song_view)
        self.simplify_chk.toggled.connect(self._refresh_song_view)
        form.addRow("Key:", self.key_value)
        form.addRow("", self.key_alt_value)
        form.addRow("Tempo:", self.tempo_value)
        form.addRow("Tuning:", self.tuning_value)
        form.addRow("Functions in:", self.function_key_combo)
        form.addRow("", self.show_functions_chk)
        form.addRow("", self.simplify_chk)
        v.addLayout(form)

        self.chord_table = QTableWidget(0, 3)
        self.chord_table.setHorizontalHeaderLabels(["Time", "Chord", "Function"])
        self.chord_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.chord_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.chord_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.chord_table.verticalHeader().setVisible(False)
        self.chord_table.verticalHeader().setDefaultSectionSize(self.chord_table.fontMetrics().height() + 8)
        ch = self.chord_table.horizontalHeader()
        ch.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        ch.setSectionResizeMode(1, QHeaderView.Stretch)
        ch.setSectionResizeMode(2, QHeaderView.Stretch)
        self.chord_table.setToolTip("Double-click a chord to play from there.")
        self.chord_table.doubleClicked.connect(self._on_chord_double_click)
        v.addWidget(self.chord_table, 1)

        # The legend doubles as the colour picker: click a swatch to change that function's colour.
        # A two-column grid, so the full names fit in the narrow side panel.
        legend = QGridLayout()
        legend.setHorizontalSpacing(6)
        legend.setVerticalSpacing(4)
        self.color_buttons: dict[int, QPushButton] = {}
        buttons = []
        for f in LEGEND_FAMILIES:
            btn = QPushButton(FAMILY_NAMES[f])
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Click to change the colour for {FAMILY_NAMES[f]} chords.")
            btn.clicked.connect(lambda _=False, f=f: self._pick_family_color(f))
            self.color_buttons[f] = btn
            buttons.append(btn)
        reset = QPushButton("Reset colours")
        reset.setToolTip("Restore the default chord-function colours.")
        reset.clicked.connect(self._reset_family_colors)
        buttons.append(reset)
        for i, btn in enumerate(buttons):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            legend.addWidget(btn, i // 2, i % 2)
        v.addLayout(legend)
        return w

    # ------------------------------------------------------- function colours
    def _update_color_buttons(self) -> None:
        size = self.fontMetrics().height()
        for f, btn in self.color_buttons.items():
            pix = QPixmap(size, size)
            pix.fill(FAMILY_COLORS[f])
            btn.setIcon(QIcon(pix))

    def _set_family_color(self, family: int, color: QColor) -> None:
        FAMILY_COLORS[family] = QColor(color)
        self._update_color_buttons()
        self._refresh_song_view()

    def _pick_family_color(self, family: int) -> None:
        color = QColorDialog.getColor(FAMILY_COLORS[family], self, f"Colour for {FAMILY_NAMES[family]} chords")
        if color.isValid():
            self._set_family_color(family, color)

    def _reset_family_colors(self) -> None:
        for f in LEGEND_FAMILIES:
            FAMILY_COLORS[f] = QColor(DEFAULT_FAMILY_COLORS[f])
        self._update_color_buttons()
        self._refresh_song_view()

    def _analysis_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.snap_chk = QCheckBox("Align chord changes to the beat")
        self.snap_chk.setToolTip("Moves each chord change onto the nearest beat when it's close to one.")
        self.min_chord_spin = QDoubleSpinBox()
        self.min_chord_spin.setRange(0.0, 4.0)
        self.min_chord_spin.setSingleStep(0.1)
        self.min_chord_spin.setDecimals(1)
        self.min_chord_spin.setSuffix(" s")
        self.min_chord_spin.setToolTip("Chords shorter than this are merged into a neighbour. Higher values give "
                                       "a simpler chart; 0 keeps every chord the model hears.")
        self.tuning_chk = QCheckBox("Compensate for the song's tuning")
        self.tuning_chk.setToolTip("Analyse chords and key relative to the measured reference pitch "
                                   "(for example A = 432 Hz) instead of A = 440 Hz.")
        self.cache_chk = QCheckBox("Remember results (re-adding a song shows them instantly)")
        clear_cache = QPushButton("Forget saved results")
        clear_cache.clicked.connect(self._clear_cache)
        self.chart_chk = QCheckBox("Save a chord chart next to each song (.chords.txt)")

        hint = QLabel("Changing these options affects songs analysed from now on. To apply them to finished songs, "
                      "select the songs and click Analyze songs.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")

        form.addRow("", self.snap_chk)
        form.addRow("Shortest chord:", self.min_chord_spin)
        form.addRow("", self.tuning_chk)
        form.addRow("", self.cache_chk)
        form.addRow("", clear_cache)
        form.addRow("", self.chart_chk)
        form.addRow("", hint)
        return w

    def _models_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.model_combos: dict[str, QComboBox] = {}
        self.model_status: dict[str, QLabel] = {}
        for task in (models.CHORDS, models.BEATS, models.KEY):
            combo = QComboBox()
            for spec in models.for_task(task):
                combo.addItem(spec.label, spec.id)
                combo.setItemData(combo.count() - 1, spec.note, Qt.ToolTipRole)
            combo.currentIndexChanged.connect(self._update_model_widgets)
            status = QLabel()
            status.setWordWrap(True)
            self.model_combos[task] = combo
            self.model_status[task] = status
            form.addRow(models.TASK_LABELS[task] + ":", combo)
            form.addRow("", status)

        self.device_combo = QComboBox()
        self.device_combo.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.setToolTip("The models are small: the CPU is fine too, only a bit slower.")
        self.download_chk = QCheckBox("Allow model downloads")
        self.download_chk.setToolTip("Only needed once per model. Uncheck for strict offline operation.")

        self.models_dir_edit = QLineEdit()
        self.models_dir_edit.editingFinished.connect(self._update_model_status)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: (self._pick_dir(self.models_dir_edit), self._update_model_status()))
        browse.setToolTip("Use models from another folder (nothing is moved).")
        move = QPushButton("Move…")
        move.clicked.connect(self._move_models)
        move.setToolTip("Move the downloaded models to a new folder and use it from now on.")
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.models_dir_edit, 1)
        dir_row.addWidget(browse)
        dir_row.addWidget(move)

        self.download_btn = QPushButton("Download selected models now")
        self.download_btn.clicked.connect(self._download_models)
        self.delete_btn = QPushButton("Delete a model…")
        self.delete_btn.clicked.connect(self._delete_model_menu)
        model_buttons = QHBoxLayout()
        model_buttons.addWidget(self.download_btn)
        model_buttons.addWidget(self.delete_btn)
        self.gpu_label = QLabel("Checking GPU…")

        form.addRow("Device:", self.device_combo)
        form.addRow("GPU:", self.gpu_label)
        form.addRow("", self.download_chk)
        form.addRow("Models folder:", dir_row)
        form.addRow("", model_buttons)
        return w

    # ------------------------------------------------------------ settings
    def _load_settings(self) -> None:
        q = self.qsettings
        d = Settings()

        def set_data(combo: QComboBox, value) -> None:
            i = combo.findData(value)
            if i >= 0:
                combo.setCurrentIndex(i)

        set_data(self.model_combos[models.CHORDS], q.value("chord_model", d.chord_model))
        set_data(self.model_combos[models.BEATS], q.value("beat_model", d.beat_model))
        set_data(self.model_combos[models.KEY], q.value("key_model", d.key_model))
        self.snap_chk.setChecked(q.value("snap_to_beats", d.snap_to_beats, bool))
        self.min_chord_spin.setValue(q.value("min_chord_seconds", d.min_chord_seconds, float))
        self.tuning_chk.setChecked(q.value("tuning_correction", d.tuning_correction, bool))
        self.cache_chk.setChecked(q.value("use_cache", d.use_cache, bool))
        self.chart_chk.setChecked(q.value("save_chart", d.save_chart, bool))
        set_data(self.device_combo, q.value("device", d.device))
        self.download_chk.setChecked(q.value("allow_download", d.allow_download, bool))
        self.models_dir_edit.setText(q.value("models_dir", str(default_models_dir())))
        self.show_functions_chk.setChecked(q.value("show_functions", True, bool))
        self.simplify_chk.setChecked(q.value("simplify", False, bool))
        self.volume.setValue(q.value("volume", 80, int))
        for f in LEGEND_FAMILIES:
            color = QColor(q.value(f"colors/{FAMILY_NAMES[f]}", DEFAULT_FAMILY_COLORS[f].name()))
            FAMILY_COLORS[f] = color if color.isValid() else QColor(DEFAULT_FAMILY_COLORS[f])
        self._update_color_buttons()
        geometry = q.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _save_settings(self) -> None:
        q = self.qsettings
        q.setValue("chord_model", self.model_combos[models.CHORDS].currentData())
        q.setValue("beat_model", self.model_combos[models.BEATS].currentData())
        q.setValue("key_model", self.model_combos[models.KEY].currentData())
        q.setValue("snap_to_beats", self.snap_chk.isChecked())
        q.setValue("min_chord_seconds", self.min_chord_spin.value())
        q.setValue("tuning_correction", self.tuning_chk.isChecked())
        q.setValue("use_cache", self.cache_chk.isChecked())
        q.setValue("save_chart", self.chart_chk.isChecked())
        q.setValue("device", self.device_combo.currentData())
        q.setValue("allow_download", self.download_chk.isChecked())
        q.setValue("models_dir", self.models_dir_edit.text())
        q.setValue("show_functions", self.show_functions_chk.isChecked())
        q.setValue("simplify", self.simplify_chk.isChecked())
        q.setValue("volume", self.volume.value())
        for f in LEGEND_FAMILIES:
            q.setValue(f"colors/{FAMILY_NAMES[f]}", FAMILY_COLORS[f].name())
        q.setValue("geometry", self.saveGeometry())

    def _models_dir(self) -> Path:
        return Path(self.models_dir_edit.text().strip() or default_models_dir())

    def _settings(self) -> Settings:
        return Settings(
            chord_model=self.model_combos[models.CHORDS].currentData(),
            beat_model=self.model_combos[models.BEATS].currentData(),
            key_model=self.model_combos[models.KEY].currentData(),
            snap_to_beats=self.snap_chk.isChecked(),
            min_chord_seconds=self.min_chord_spin.value(),
            tuning_correction=self.tuning_chk.isChecked(),
            use_cache=self.cache_chk.isChecked(),
            save_chart=self.chart_chk.isChecked(),
            device=self.device_combo.currentData(),
            models_dir=self._models_dir(),
            cache_dir=default_cache_dir(),
            allow_download=self.download_chk.isChecked(),
        )

    def _update_model_widgets(self) -> None:
        for task, combo in self.model_combos.items():
            combo.setToolTip(models.get(combo.currentData(), task).note)
        self._update_model_status()

    def _update_model_status(self) -> None:
        md = self._models_dir()
        for task, combo in self.model_combos.items():
            spec = models.get(combo.currentData(), task)
            if not spec.needs_download:
                text = "Built in, nothing to download."
            elif model_store.is_installed(md, spec):
                text = f"Downloaded ({model_store.installed_size(md, spec) / 2**20:.0f} MB)."
            elif model_store.is_partial(md, spec):
                text = f"Partly downloaded (~{spec.size_mb:.0f} MB total)."
            else:
                text = f"Not downloaded (~{spec.size_mb:.0f} MB)."
            self.model_status[task].setText(f"{text} {spec.note}")
        if hasattr(self, "delete_btn"):
            self._update_buttons()

    # --------------------------------------------------------------- songs
    def _add_paths(self, paths: list[Path]) -> None:
        files = audio_io.collect(paths)
        existing = {str(self.table.item(r, COL_FILE).data(PATH_ROLE)).lower() for r in range(self.table.rowCount())}
        s = self._settings()
        added = 0
        for f in files:
            if str(f).lower() in existing:
                continue
            existing.add(str(f).lower())
            row = self._add_row(f)
            added += 1
            cached = load_cached(f, s) if s.use_cache else None
            if cached:
                self._set_result(row, cached)
        if paths and not files:
            self._log("No audio files found in what was added.")
        elif added and self.table.currentRow() < 0:
            self.table.selectRow(self.table.rowCount() - added)
        self._update_buttons()

    def _add_row(self, path: Path) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        item = QTableWidgetItem(path.name)
        item.setData(PATH_ROLE, str(path))
        item.setToolTip(str(path))
        self.table.setItem(row, COL_FILE, item)
        for col in (COL_STATUS, COL_BPM, COL_KEY, COL_CAMELOT, COL_TUNING):
            self.table.setItem(row, col, QTableWidgetItem("Queued" if col == COL_STATUS else ""))
        return row

    def _path(self, row: int) -> Path:
        return Path(self.table.item(row, COL_FILE).data(PATH_ROLE))

    def _result(self, row: int | None) -> Analysis | None:
        if row is None or not 0 <= row < self.table.rowCount():
            return None
        return self.table.item(row, COL_FILE).data(RESULT_ROLE)

    def _set_result(self, row: int, r: Analysis) -> None:
        self.table.item(row, COL_FILE).setData(RESULT_ROLE, r)
        self.table.item(row, COL_STATUS).setText("Done")
        self.table.item(row, COL_STATUS).setToolTip(f"Analysed in {r.elapsed:.1f} s" if r.elapsed else "")
        self.table.item(row, COL_BPM).setText(rhythm.format_bpm(r.bpm) if r.bpm else "–")
        self.table.item(row, COL_BPM).setToolTip(r.tempo.text())
        self.table.item(row, COL_KEY).setText(r.key.name)
        self.table.item(row, COL_KEY).setToolTip(f"{r.key_confidence:.0%} confident")
        self.table.item(row, COL_CAMELOT).setText(r.key.camelot)
        self.table.item(row, COL_CAMELOT).setToolTip(f"Open Key: {r.key.open_key}")
        self.table.item(row, COL_TUNING).setText(rhythm.short_tuning(r.tuning_cents))
        self.table.item(row, COL_TUNING).setToolTip(rhythm.describe_tuning(r.tuning_cents))
        if row == self.current_row:
            self._show_song(row)

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _remove_selected(self) -> None:
        rows = self._selected_rows()
        self._show_song(None)  # row numbers shift; the selection picks the new current song
        for row in reversed(rows):
            self.table.removeRow(row)
        self._on_selection()

    def _clear(self) -> None:
        self._show_song(None)
        self.table.setRowCount(0)
        self._on_selection()

    def _show_in_folder(self) -> None:
        rows = self._selected_rows()
        if rows:
            subprocess.Popen(["explorer", "/select,", str(self._path(rows[0]))])

    def _pick_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(audio_io.AUDIO_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add songs", self.qsettings.value("last_dir", ""),
                                                f"Audio files ({exts});;All files (*)")
        if files:
            self.qsettings.setValue("last_dir", str(Path(files[0]).parent))
            self._add_paths([Path(f) for f in files])

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of songs", self.qsettings.value("last_dir", ""))
        if folder:
            self.qsettings.setValue("last_dir", folder)
            self._add_paths([Path(folder)])

    def _pick_dir(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text())
        if folder:
            edit.setText(folder)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        self._add_paths(paths)

    def _export(self) -> None:
        row = self.current_row
        r = self._result(row)
        if r is None:
            return
        stem = Path(r.path).stem
        start = str(Path(self.qsettings.value("export_dir", str(Path(r.path).parent))) / f"{stem}.chords.txt")
        filters = "Chord chart (*.txt);;Chord labels, MIREX format (*.lab);;Full analysis (*.json)"
        file, chosen = QFileDialog.getSaveFileName(self, "Export chords", start, filters)
        if not file:
            return
        path = Path(file)
        self.qsettings.setValue("export_dir", str(path.parent))
        kind = path.suffix.lower()
        if kind not in (".txt", ".lab", ".json"):  # no or unusual extension: go by the chosen filter
            kind = ".lab" if "lab" in chosen else ".json" if "json" in chosen else ".txt"
            path = path.with_name(path.name + kind)
        try:
            if kind == ".lab":
                path.write_text(r.lab(), encoding="utf-8")
            elif kind == ".json":
                path.write_text(r.to_json(), encoding="utf-8")
            else:
                path.write_text(r.chart(self._function_key(row), self.show_functions_chk.isChecked()),
                                encoding="utf-8")
            self._log(f"Saved {path}")
        except OSError as e:
            QMessageBox.warning(self, "Export chords", f"Could not save the file: {e}")

    def _clear_cache(self) -> None:
        folder = default_cache_dir()
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        self._log("Saved results forgotten.")

    # ------------------------------------------------- current song view
    def _on_selection(self) -> None:
        rows = self._selected_rows()
        row = rows[0] if rows else None
        if row != self.current_row:
            self._show_song(row)
        self._update_buttons()

    def _function_key(self, row: int | None) -> Key | None:
        r = self._result(row)
        if r is None:
            return None
        override = self.table.item(row, COL_FILE).data(KEY_ROLE)
        return Key.from_index(override) if override is not None else r.key

    def _show_song(self, row: int | None) -> None:
        """Make row the current song: load it into the player and show its chords."""
        if row != self.current_row:
            self._unload_player()
        self.current_row = row
        r = self._result(row)
        if row is None:
            self.now_playing.setText("")
        else:
            self.now_playing.setText(self._path(row).name)
        if r is None:
            msg = ("Add songs and click Analyze songs." if row is None
                   else "This song hasn't been analysed yet. Click Analyze songs.")
            self.chord_display.set_song([], message=msg)
            self.timeline.set_song([])
            self.chord_table.setRowCount(0)
            for label in (self.key_value, self.tempo_value, self.tuning_value):
                label.setText("–")
            self.key_alt_value.setText("")
            self.summary_label.setText("")
            self.function_key_combo.setCurrentIndex(0)
            self.function_key_combo.setEnabled(False)
            return

        key = r.key
        self.key_value.setText(f"{key.name}   ·   {key.camelot}")
        self.key_value.setToolTip(f"Camelot {key.camelot}, Open Key {key.open_key}, "
                                  f"{r.key_confidence:.0%} confident")
        others = ", ".join(f"{k.name} ({k.camelot}) {p:.0%}" for k, p in r.key_candidates(3)[1:] if p >= 0.01)
        self.key_alt_value.setText(f"{r.key_confidence:.0%} confident" + (f"; also possible: {others}" if others else ""))
        meter = f", {r.meter_text}" if r.meter else ""
        self.tempo_value.setText(r.tempo.text() + meter)
        self.tuning_value.setText(rhythm.describe_tuning(r.tuning_cents))
        self.summary_label.setText(f"{rhythm.format_bpm(r.bpm)} BPM · {key.short} ({key.camelot}) · "
                                   f"{rhythm.short_tuning(r.tuning_cents)}" if r.bpm else "")
        override = self.table.item(row, COL_FILE).data(KEY_ROLE)
        self.function_key_combo.setEnabled(True)
        self.function_key_combo.setItemText(0, f"Detected key ({key.name})")
        i = self.function_key_combo.findData(override) if override is not None else 0
        self.function_key_combo.setCurrentIndex(max(0, i))
        self._refresh_song_view()

    def _refresh_song_view(self) -> None:
        row = self.current_row
        r = self._result(row)
        show = self.show_functions_chk.isChecked()
        self.chord_display.show_functions = show
        self.timeline.show_functions = show
        if r is None:
            self.chord_display.update()
            self.timeline.update()
            return
        key = self._function_key(row)
        items = build_items(r.chords, key, self.simplify_chk.isChecked())
        self.chord_display.set_song(items, r.beats, r.downbeats, r.meter)
        self.timeline.set_song(items, r.beats, r.downbeats, r.meter)
        self.chord_table.setRowCount(0)
        rows = [it for it in items if it.symbol != "N.C."]
        self.chord_table.setRowCount(len(rows))
        for i, it in enumerate(rows):
            t = QTableWidgetItem(_fmt_seconds(it.start))
            t.setData(Qt.UserRole, it.start)
            name = QTableWidgetItem(it.symbol)
            func = QTableWidgetItem(it.numeral if show else "")
            func.setForeground(QBrush(FAMILY_COLORS[it.family]))
            font = name.font()
            font.setBold(True)
            name.setFont(font)
            for col, item in enumerate((t, name, func)):
                self.chord_table.setItem(i, col, item)
        self._chord_starts = [it.start for it in rows]
        self._chord_row = -1
        self._on_tick(force=True)

    def _on_function_key(self) -> None:
        row = self.current_row
        if row is None:
            return
        self.table.item(row, COL_FILE).setData(KEY_ROLE, self.function_key_combo.currentData())
        self._refresh_song_view()

    def _on_chord_double_click(self, index) -> None:
        start = self.chord_table.item(index.row(), 0).data(Qt.UserRole)
        if start is not None:
            self._seek_seconds(float(start))
            self._play()

    # -------------------------------------------------------------- player
    def _ensure_player(self):
        if self.player is None:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self.audio_out = QAudioOutput(self)
            self.audio_out.setVolume(self.volume.value() / 100)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.audio_out)
            self.player.durationChanged.connect(lambda d: self.seek.setRange(0, d))
            self.player.positionChanged.connect(self._on_position)
            self.player.playbackStateChanged.connect(self._on_playback_state)
            self.player.errorOccurred.connect(lambda _e, msg: self._log(f"Playback error: {msg}"))
        return self.player

    def _set_volume(self, value: int) -> None:
        if self.player is not None:
            self.audio_out.setVolume(value / 100)

    def _load_current(self) -> bool:
        """Make sure the current song is the player's source."""
        if self.current_row is None:
            return False
        path = str(self._path(self.current_row))
        player = self._ensure_player()
        if player.source().toLocalFile() != QUrl.fromLocalFile(path).toLocalFile():
            player.setSource(QUrl.fromLocalFile(path))
        return True

    def _unload_player(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())
        self.seek.setRange(0, 0)
        self.time_label.setText("0:00 / 0:00")
        self.chord_display.set_time(0.0)
        self.timeline.set_time(0.0)

    def _play(self) -> None:
        if self._load_current():
            self.player.play()

    def _play_row(self, row: int) -> None:
        if row != self.current_row:
            self.table.selectRow(row)
        self._play()

    def _toggle_play(self) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        if self.player and self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self._play()

    def _seek_ms(self, ms: int) -> None:
        if self._load_current():
            self.player.setPosition(ms)
            self._on_tick(force=True)

    def _seek_seconds(self, t: float) -> None:
        self._seek_ms(int(t * 1000))
        self.seek.setValue(int(t * 1000))

    def _skip(self, seconds: float) -> None:
        if self.player is not None:
            self._seek_seconds(max(0.0, self.player.position() / 1000 + seconds))

    def _on_position(self, pos: int) -> None:
        if not self.seek.isSliderDown():
            self.seek.setValue(pos)
        self.time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(self.player.duration())}")
        if not self.tick.isActive():
            self._on_tick()

    def _on_playback_state(self, state) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        playing = state == QMediaPlayer.PlayingState
        self.play_btn.setText("❚❚ Pause" if playing else "▶ Play")
        if playing:
            self.tick.start()
        else:
            self.tick.stop()
            self._on_tick(force=True)

    def _display_time(self, force: bool) -> float:
        """Playback position in seconds, extrapolated smoothly between the player's coarse updates."""
        from PySide6.QtMultimedia import QMediaPlayer
        reported = self.player.position() / 1000
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        if force or not playing or not self._clock.isValid():
            t = reported
        else:
            t = self._clock_pos + self._clock.elapsed() / 1000 * self.player.playbackRate()
            error = reported - t
            if abs(error) > 0.25:      # seek, buffering stall, etc.: jump to the real position
                t = reported
            elif reported != self._last_reported:
                # Drift gently toward each fresh report, but never backwards (that reads as a judder).
                t = max(self._clock_pos, t + error * 0.2)
        self._last_reported = reported
        self._clock_pos = t
        self._clock.start()
        return t

    def _on_tick(self, force: bool = False) -> None:
        """Move the chord display along with the playback position (~60 times a second)."""
        if self.player is None or self.current_row is None:
            return
        t = self._display_time(force)
        self.chord_display.set_time(t)
        self.timeline.set_time(t)
        starts = getattr(self, "_chord_starts", [])
        i = bisect.bisect_right(starts, t + 0.001) - 1
        if i != self._chord_row or force:
            self._chord_row = i
            self.chord_table.clearSelection()
            if i >= 0:
                self.chord_table.selectRow(i)
                self.chord_table.scrollToItem(self.chord_table.item(i, 0), QAbstractItemView.PositionAtCenter)

    # ----------------------------------------------------------------- run
    def _busy(self) -> bool:
        return any(t is not None and t.isRunning() for t in (self.worker, self.helper))

    def _update_buttons(self) -> None:
        busy = self._busy()
        rows = self._selected_rows()
        has_rows = self.table.rowCount() > 0
        self.start_btn.setEnabled(not busy and has_rows)
        self.cancel_btn.setEnabled(self.worker is not None and self.worker.isRunning())
        self.remove_btn.setEnabled(not busy and bool(rows))
        self.clear_btn.setEnabled(not busy and has_rows)
        self.export_btn.setEnabled(self._result(self.current_row) is not None)
        self.folder_btn.setEnabled(bool(rows))
        self.play_btn.setEnabled(self.current_row is not None)
        for w in (self.tabs.widget(1), self.tabs.widget(2)):
            w.setEnabled(not busy)
        md = self._models_dir()
        missing = [s for s in self._settings().specs() if not model_store.is_installed(md, s)]
        self.download_btn.setEnabled(not busy and bool(missing))
        any_installed = any(s.needs_download and model_store.model_dir(md, s).is_dir() for s in models.MODELS)
        self.delete_btn.setEnabled(not busy and any_installed)

    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _set_status(self, row: int, text: str) -> None:
        self.table.item(row, COL_STATUS).setText(text)

    def _start(self) -> None:
        s = self._settings()
        pending = [r for r in range(self.table.rowCount()) if self.table.item(r, COL_STATUS).text() != "Done"]
        if not pending:
            pending = self._selected_rows()
            s.use_cache = False  # the user asked to analyse finished songs again
            if not pending:
                QMessageBox.information(self, "Chordfinder", "All songs are analysed. Select songs to analyse "
                                                             "them again with the current options.")
                return
        jobs = []
        for row in pending:
            jobs.append((row, self._path(row)))
            self._set_status(row, "Queued")
        self._save_settings()
        self.total_bar.setRange(0, len(jobs))
        self.total_bar.setValue(0)
        self.job_bar.setValue(0)

        self.worker = Worker(self.pipeline, jobs, s)
        self.worker.log.connect(self._log)
        self.worker.job_started.connect(self._on_started)
        self.worker.job_progress.connect(self._on_progress)
        self.worker.job_done.connect(self._on_done)
        self.worker.job_failed.connect(self._on_failed)
        self.worker.job_cancelled.connect(lambda r: self._set_status(r, "Cancelled"))
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._update_buttons()

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.current_label.setText("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def _on_started(self, row: int) -> None:
        name = self._path(row).name
        self.current_label.setText(f"Analyzing: {name[:70]}{'…' if len(name) > 70 else ''}")
        self._set_status(row, "Starting…")
        self.job_bar.setValue(0)
        self.table.scrollToItem(self.table.item(row, COL_FILE))

    def _on_progress(self, row: int, fraction: float, stage: str) -> None:
        pct = int(fraction * 100)
        self.job_bar.setValue(pct)
        self._set_status(row, f"{stage} {pct}%")

    def _on_done(self, row: int, result: Analysis) -> None:
        self._set_result(row, result)
        self.total_bar.setValue(self.total_bar.value() + 1)
        if self.chart_chk.isChecked():
            self._log(f"Chord chart saved to {chart_path(Path(result.path)).name}")
        self._update_buttons()

    def _on_failed(self, row: int, message: str) -> None:
        self._set_status(row, "Failed")
        self.table.item(row, COL_STATUS).setToolTip(message)
        self.table.item(row, COL_BPM).setText("")
        self.total_bar.setValue(self.total_bar.value() + 1)

    def _on_finished(self) -> None:
        self.current_label.setText("Idle")
        self._log("Finished.")
        self._update_model_status()
        self._update_buttons()

    # ------------------------------------------------------- model helpers
    def _download_models(self) -> None:
        s = self._settings()
        specs = [spec for spec in s.specs() if not model_store.is_installed(s.models_dir, spec)]
        if not specs:
            return
        self._save_settings()
        self.current_label.setText("Downloading models…")
        self.job_bar.setValue(0)

        def job():
            for spec in specs:
                model_store.download(s.models_dir, spec, self._log_threadsafe, self._progress_threadsafe)
            return "The selected models are downloaded. Chordfinder can now run fully offline."

        self._run_helper(job, on_done=self._on_helper_done, on_failed=self._on_helper_failed)

    def _delete_model_menu(self) -> None:
        md = self._models_dir()
        menu = QMenu(self)
        for spec in models.MODELS:
            if spec.needs_download and model_store.model_dir(md, spec).is_dir():
                size = model_store.installed_size(md, spec) / 2**20
                menu.addAction(f"{models.TASK_LABELS[spec.task]}: {spec.label} ({size:.0f} MB)",
                               lambda sp=spec: self._delete_model(sp))
        menu.exec(self.delete_btn.mapToGlobal(self.delete_btn.rect().bottomLeft()))

    def _delete_model(self, spec) -> None:
        md = self._models_dir()
        size_mb = model_store.installed_size(md, spec) / 2**20
        if QMessageBox.question(self, "Delete model",
                                f"Delete {spec.label} ({size_mb:.0f} MB) from\n{model_store.model_dir(md, spec)}?\n\n"
                                "You can download it again later.") != QMessageBox.Yes:
            return
        self.pipeline.release()  # loaded models keep files open on Windows
        try:
            model_store.delete_model(md, spec)
            self._log(f"Deleted {spec.label} ({size_mb:.0f} MB freed).")
        except OSError as e:
            QMessageBox.warning(self, "Delete model", f"Could not delete the model: {e}")
        self._update_model_status()

    def _move_models(self) -> None:
        src = self._models_dir()
        entries = model_store.installed(src)
        if not entries:
            QMessageBox.information(self, "Move models", f"No downloaded models found in\n{src}")
            return
        folder = QFileDialog.getExistingDirectory(self, "Move models to", str(src.parent))
        if not folder:
            return
        dst = Path(folder)
        size_mb = model_store.size_of(entries) / 2**20
        names = ", ".join(e.name for e in entries)
        if QMessageBox.question(
                self, "Move models",
                f"Move {size_mb:.0f} MB of models ({names})\n\nfrom: {src}\nto: {dst}") != QMessageBox.Yes:
            return

        self.pipeline.release()  # loaded models keep files open, which blocks moving on Windows
        self._log(f"Moving models to {dst}…")
        self.current_label.setText("Moving models…")
        self.job_bar.setValue(0)

        def job():
            model_store.move_models(src, dst, self._progress_threadsafe)
            return dst

        self._run_helper(job, on_done=self._on_models_moved, on_failed=self._on_helper_failed)

    def _on_models_moved(self, dst: Path) -> None:
        self.models_dir_edit.setText(str(dst))
        self._save_settings()
        self.current_label.setText("Idle")
        self._log(f"Models moved. Models folder is now {dst}")
        self._update_model_status()

    def _on_helper_done(self, msg) -> None:
        self.current_label.setText("Idle")
        self._log(str(msg))
        self._update_model_status()

    def _on_helper_failed(self, err: str) -> None:
        self.current_label.setText("Idle")
        self.job_bar.setValue(0)
        self._update_model_status()
        QMessageBox.warning(self, "Chordfinder", err)

    def _log_threadsafe(self, text: str) -> None:
        if self.helper:
            self.helper.message.emit(text)

    def _progress_threadsafe(self, fraction: float) -> None:
        if self.helper:
            self.helper.progress.emit(fraction)

    def _run_helper(self, fn, on_done=None, on_failed=None) -> None:
        self.helper = FunctionThread(fn)
        self.helper.finished_ok.connect(on_done or (lambda msg: self._log(str(msg))))
        self.helper.message.connect(self._log)
        self.helper.progress.connect(lambda f: self.job_bar.setValue(int(f * 100)))
        self.helper.failed.connect(lambda err: self._log(f"Error: {err}"))
        if on_failed:
            self.helper.failed.connect(on_failed)
        self.helper.finished.connect(self._update_buttons)
        self.helper.start()
        self._update_buttons()

    def _probe_gpu(self) -> None:
        self.gpu_thread = FunctionThread(_probe_gpu)
        self.gpu_thread.finished_ok.connect(self._on_gpu)
        self.gpu_thread.failed.connect(lambda err: self.gpu_label.setText(f"Unavailable ({err})"))
        self.gpu_thread.start()

    def _on_gpu(self, name: str | None) -> None:
        if name:
            self.gpu_label.setText(name)
            self._log(f"GPU ready: {name}")
        else:
            self.gpu_label.setText("No CUDA GPU detected – using CPU")
            self.device_combo.setCurrentIndex(self.device_combo.findData("cpu"))
            self._log("No CUDA GPU detected; using the CPU.")

    def closeEvent(self, event) -> None:
        if self.worker and self.worker.isRunning():
            if QMessageBox.question(self, "Chordfinder", "Songs are still being analysed. Quit anyway?") \
                    != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancel()
            self.worker.wait(10000)
        if self.player:
            self.player.stop()
        self._save_settings()
        event.accept()
