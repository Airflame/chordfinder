"""Downloading, inspecting, deleting and relocating the local model files.

The models folder contains one subfolder per task (see MODEL_SUBDIRS), and in
it one folder per model. Only the task subfolders are ever moved, so pointing
the models folder at a directory that also holds unrelated files is safe.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Callable

from .models import CHORDS, BEATS, KEY, ModelSpec

MODEL_SUBDIRS = (CHORDS, BEATS, KEY)
_CHUNK = 16 * 1024 * 1024
_DOWNLOAD_CHUNK = 256 * 1024
_USER_AGENT = "Chordfinder model downloader"


class MoveError(Exception):
    """A move that was refused or failed; the message is meant for the user."""


class DownloadError(Exception):
    """A download that failed; the message is meant for the user."""


# ------------------------------------------------------------ single models
def model_dir(models_dir: Path, spec: ModelSpec) -> Path:
    return models_dir / spec.subdir / spec.id


def file_path(models_dir: Path, spec: ModelSpec, name: str | None = None) -> Path:
    """Path of one of the model's files (the first one by default)."""
    return model_dir(models_dir, spec) / (name or spec.files[0].name)


def is_installed(models_dir: Path, spec: ModelSpec) -> bool:
    """True when every file of the model is present with its full size."""
    if not spec.needs_download:
        return True
    d = model_dir(models_dir, spec)
    return all((d / f.name).is_file() and (d / f.name).stat().st_size == f.size for f in spec.files)


def is_partial(models_dir: Path, spec: ModelSpec) -> bool:
    return spec.needs_download and model_dir(models_dir, spec).is_dir() and not is_installed(models_dir, spec)


def installed_size(models_dir: Path, spec: ModelSpec) -> int:
    d = model_dir(models_dir, spec)
    return size_of([d]) if d.is_dir() else 0


def delete_model(models_dir: Path, spec: ModelSpec) -> None:
    d = model_dir(models_dir, spec)
    if d.is_dir():
        shutil.rmtree(d)
    try:
        d.parent.rmdir()  # the task folder, if this was its last model
    except OSError:
        pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(models_dir: Path, spec: ModelSpec, log: Callable[[str], None] = print,
             progress: Callable[[float], None] = lambda f: None,
             is_cancelled: Callable[[], bool] = lambda: False) -> None:
    """Download a model's files (resumes partial downloads, verifies hashes)."""
    if not spec.needs_download:
        return
    d = model_dir(models_dir, spec)
    d.mkdir(parents=True, exist_ok=True)
    total = sum(f.size for f in spec.files) or 1
    done = 0
    log(f"Downloading {spec.label} ({spec.size_mb:.0f} MB)…")
    for f in spec.files:
        target = d / f.name
        if target.is_file() and target.stat().st_size == f.size:
            done += f.size
            progress(done / total)
            continue
        part = target.with_name(target.name + ".part")
        _fetch(f.url, part, f.size, lambda n, base=done: progress(min(1.0, (base + n) / total)), is_cancelled)
        digest = sha256_of(part)
        if digest != f.sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(f"{f.name} failed its integrity check (got SHA-256 {digest[:12]}…). "
                                "The file was deleted; please try again.")
        os.replace(part, target)
        done += f.size
        progress(done / total)
    progress(1.0)


def _fetch(url: str, part: Path, size: int, progress: Callable[[int], None],
           is_cancelled: Callable[[], bool]) -> None:
    """Download url into part, continuing where an earlier attempt stopped."""
    have = part.stat().st_size if part.is_file() else 0
    if have >= size:  # complete or oversized leftovers: start over
        part.unlink(missing_ok=True)
        have = 0
    headers = {"User-Agent": _USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
            if have and getattr(resp, "status", None) != 206:  # server ignored the range request
                have = 0
            with open(part, "ab" if have else "wb") as out:
                n = have
                while chunk := resp.read(_DOWNLOAD_CHUNK):
                    if is_cancelled():
                        from .analyzer import Cancelled
                        raise Cancelled()
                    out.write(chunk)
                    n += len(chunk)
                    progress(n)
    except OSError as e:  # URLError and HTTPError are OSErrors
        raise DownloadError(f"Could not download {url}: {getattr(e, 'reason', e)}. "
                            "Check the internet connection and try again.") from e


# ------------------------------------------------------------ whole folder
def installed(models_dir: Path) -> list[Path]:
    """The task subfolders present in models_dir."""
    return [models_dir / name for name in MODEL_SUBDIRS if (models_dir / name).is_dir()]


def size_of(paths: list[Path]) -> int:
    return sum(f.stat().st_size for p in paths for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def move_models(src: Path, dst: Path, progress: Callable[[float], None] = lambda f: None) -> None:
    """Move the model subfolders of src into dst.

    Same drive: renames, which is instant. Different drive: copies everything,
    and deletes the originals only once the whole copy succeeded.
    """
    src, dst = src.resolve(), dst.resolve()
    entries = installed(src)
    if not entries:
        raise MoveError(f"No downloaded models found in {src}.")
    if dst == src:
        raise MoveError("The models are already in that folder.")
    if dst.is_relative_to(src):
        raise MoveError("The new location can't be inside the current models folder.")
    conflicts = [dst / e.name for e in entries if (dst / e.name).exists()]
    if conflicts:
        raise MoveError(f"{conflicts[0]} already exists. If it holds models, "
                        "select that folder with Browse… instead.")

    dst.mkdir(parents=True, exist_ok=True)
    if _same_device(src, dst):
        _rename_all(entries, dst)
    else:
        total = size_of(entries)
        if shutil.disk_usage(dst).free < total:
            raise MoveError(f"Not enough free space in {dst} ({total / 2**20:.0f} MB needed).")
        _copy_all(entries, dst, total, progress)
        for e in entries:
            shutil.rmtree(e)
    progress(1.0)

    try:
        src.rmdir()  # only succeeds if nothing else is left in it
    except OSError:
        pass


def _same_device(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def _rename_all(entries: list[Path], dst: Path) -> None:
    moved: list[Path] = []
    try:
        for e in entries:
            os.replace(e, dst / e.name)
            moved.append(e)
    except OSError as err:
        for e in moved:  # put back what was already moved
            os.replace(dst / e.name, e)
        raise MoveError(f"Could not move {e.name}: {err}. Close any program using the models and retry.")


def _copy_all(entries: list[Path], dst: Path, total: int, progress: Callable[[float], None]) -> None:
    done = 0
    created: list[Path] = []
    try:
        for e in entries:
            created.append(dst / e.name)
            for f in e.rglob("*"):
                target = dst / f.relative_to(e.parent)
                if f.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(f, "rb") as fin, open(target, "wb") as fout:
                    while chunk := fin.read(_CHUNK):
                        fout.write(chunk)
                        done += len(chunk)
                        progress(done / total if total else 1.0)
                shutil.copystat(f, target)
    except OSError as err:
        for c in created:
            shutil.rmtree(c, ignore_errors=True)
        raise MoveError(f"Copy failed: {err}. The models were left in their original location.")
