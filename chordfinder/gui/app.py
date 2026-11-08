"""GUI entry point."""

from __future__ import annotations

import sys


def use_own_taskbar_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("chordfinder.app")
    except (AttributeError, OSError):
        pass


def give_streams_to_windowless_process() -> None:
    """Under pythonw.exe sys.stdout/stderr are None, and any library that prints
    (warnings, progress bars) crashes with "'NoneType' object has no attribute
    'write'". Send that output to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    import os
    from pathlib import Path
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Chordfinder"
    try:
        base.mkdir(parents=True, exist_ok=True)
        stream = open(base / "chordfinder.log", "w", encoding="utf-8", buffering=1, errors="replace")
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def main() -> int:
    give_streams_to_windowless_process()

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from .main_window import ICON_PATH, MainWindow

    use_own_taskbar_identity()
    app = QApplication(sys.argv)
    app.setApplicationName("Chordfinder")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow(files=sys.argv[1:])
    window.show()
    return app.exec()
