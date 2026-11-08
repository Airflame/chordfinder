"""Background thread that analyses the queued songs."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..analyzer import Callbacks, Cancelled, Pipeline, Settings


class Worker(QThread):
    log = Signal(str)
    job_started = Signal(int)
    job_progress = Signal(int, float, str)
    job_done = Signal(int, object)        # row, Analysis
    job_failed = Signal(int, str)
    job_cancelled = Signal(int)

    def __init__(self, pipeline: Pipeline, jobs: list[tuple[int, Path]], settings: Settings):
        super().__init__()
        self.pipeline = pipeline
        self.jobs = jobs
        self.settings = settings
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        for row, path in self.jobs:
            if self._cancel.is_set():
                break
            self.job_started.emit(row)
            callbacks = Callbacks(
                log=self.log.emit,
                progress=lambda f, stage, r=row: self.job_progress.emit(r, f, stage),
                is_cancelled=self._cancel.is_set,
            )
            try:
                result = self.pipeline.analyze(path, self.settings, callbacks)
                self.job_done.emit(row, result)
            except Cancelled:
                self.job_cancelled.emit(row)
                self.log.emit("Cancelled.")
                break
            except Exception as e:
                self.log.emit(traceback.format_exc())
                self.job_failed.emit(row, str(e) or type(e).__name__)
