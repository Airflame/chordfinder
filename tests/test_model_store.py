import hashlib
import http.server
import threading
from dataclasses import replace

import pytest

from chordfinder import model_store, models
from chordfinder.model_store import DownloadError, MoveError, move_models
from chordfinder.models import ModelFile


def test_every_model_task_has_a_default_and_pinned_files():
    for task in (models.CHORDS, models.BEATS, models.KEY):
        assert models.get(models.DEFAULTS[task]).task == task
    for spec in models.MODELS:
        for f in spec.files:
            assert len(f.sha256) == 64 and f.size > 0 and f.url.startswith("https://")


def test_unknown_model_id_falls_back_to_task_default():
    assert models.get("nope", models.BEATS).id == models.DEFAULTS[models.BEATS]
    assert models.get("skey", models.CHORDS).id == models.DEFAULTS[models.CHORDS]  # wrong task


def test_models_without_files_are_always_installed(tmp_path):
    assert model_store.is_installed(tmp_path, models.get("librosa-beats"))
    assert model_store.is_installed(tmp_path, models.get("profile-key"))


def test_install_status_and_delete(tmp_path):
    spec = models.get("btc-majmin")
    d = model_store.model_dir(tmp_path, spec)
    assert d == tmp_path / "chords" / "btc-majmin"
    assert not model_store.is_installed(tmp_path, spec)
    d.mkdir(parents=True)
    (d / spec.files[0].name).write_bytes(b"x" * 100)
    assert not model_store.is_installed(tmp_path, spec)   # wrong size: partial download
    assert model_store.is_partial(tmp_path, spec)
    (d / spec.files[0].name).write_bytes(b"x" * spec.files[0].size)
    assert model_store.is_installed(tmp_path, spec)
    model_store.delete_model(tmp_path, spec)
    assert not d.exists() and not (tmp_path / "chords").exists()


# ------------------------------------------------------------------ download
@pytest.fixture
def server(tmp_path):
    """A local HTTP server (without Range support, like some CDNs)."""
    root = tmp_path / "www"
    root.mkdir()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(root), **kw)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.RequestHandlerClass.log_message = lambda *a: None
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()


def _spec(payload: bytes, url: str, sha: str | None = None):
    f = ModelFile("weights.pt", url, sha or hashlib.sha256(payload).hexdigest(), len(payload))
    return replace(models.get("skey"), files=(f,))


def test_download_verifies_and_installs(server, tmp_path):
    root, base = server
    payload = bytes(range(256)) * 4000
    (root / "w.pt").write_bytes(payload)
    spec = _spec(payload, base + "w.pt")
    seen = []
    model_store.download(tmp_path / "models", spec, log=lambda m: None, progress=seen.append)
    assert model_store.is_installed(tmp_path / "models", spec)
    assert model_store.file_path(tmp_path / "models", spec).read_bytes() == payload
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_partial_download_is_restarted_when_server_ignores_range(server, tmp_path):
    root, base = server
    payload = b"abc" * 50000
    (root / "w.pt").write_bytes(payload)
    spec = _spec(payload, base + "w.pt")
    part = model_store.file_path(tmp_path, spec).with_name("weights.pt.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"GARBAGE")
    model_store.download(tmp_path, spec, log=lambda m: None)
    assert model_store.file_path(tmp_path, spec).read_bytes() == payload
    assert not part.exists()


def test_corrupt_download_is_rejected_and_deleted(server, tmp_path):
    root, base = server
    (root / "w.pt").write_bytes(b"tampered")
    spec = _spec(b"tampered", base + "w.pt", sha="0" * 64)
    with pytest.raises(DownloadError, match="integrity"):
        model_store.download(tmp_path, spec, log=lambda m: None)
    assert not model_store.is_installed(tmp_path, spec)
    assert not any(model_store.model_dir(tmp_path, spec).iterdir())


def test_unreachable_url_gives_readable_error(server, tmp_path):
    _, base = server
    spec = _spec(b"x", base + "missing.pt")
    with pytest.raises(DownloadError, match="Could not download"):
        model_store.download(tmp_path, spec, log=lambda m: None)


# ---------------------------------------------------------------------- move
@pytest.fixture
def store(tmp_path):
    src = tmp_path / "models"
    (src / "chords" / "m").mkdir(parents=True)
    (src / "chords" / "m" / "model.pt").write_bytes(b"w" * 1000)
    (src / "key").mkdir()
    (src / "key" / "weights").write_bytes(b"n" * 500)
    return src


def assert_moved(src, dst):
    assert (dst / "chords" / "m" / "model.pt").read_bytes() == b"w" * 1000
    assert (dst / "key" / "weights").read_bytes() == b"n" * 500
    assert not (src / "chords").exists() and not (src / "key").exists()


def test_move_same_drive(store, tmp_path):
    dst = tmp_path / "new" / "place"
    move_models(store, dst)
    assert_moved(store, dst)
    assert not store.exists()


def test_move_across_drives_copies_with_progress(store, tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "_same_device", lambda a, b: False)
    seen = []
    dst = tmp_path / "other"
    move_models(store, dst, seen.append)
    assert_moved(store, dst)
    assert seen[-1] == 1.0 and seen == sorted(seen)


def test_unrelated_files_are_left_alone(store, tmp_path):
    (store / "notes.txt").write_text("keep me")
    move_models(store, tmp_path / "dst")
    assert (store / "notes.txt").read_text() == "keep me"


def test_failed_copy_rolls_back(store, tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "_same_device", lambda a, b: False)

    def boom(*_):
        raise OSError("disk full")

    monkeypatch.setattr(model_store.shutil, "copystat", boom)
    dst = tmp_path / "dst"
    with pytest.raises(MoveError):
        move_models(store, dst)
    assert (store / "chords" / "m" / "model.pt").exists()
    assert not (dst / "chords").exists()


@pytest.mark.parametrize("make_dst, message", [
    (lambda src, tmp: src, "already in that folder"),
    (lambda src, tmp: src / "sub", "inside"),
])
def test_refused_destinations(store, tmp_path, make_dst, message):
    with pytest.raises(MoveError, match=message):
        move_models(store, make_dst(store, tmp_path))


def test_refuses_to_overwrite_existing_models(store, tmp_path):
    (tmp_path / "dst" / "key").mkdir(parents=True)
    with pytest.raises(MoveError, match="already exists"):
        move_models(store, tmp_path / "dst")
    assert (store / "key" / "weights").exists()


def test_nothing_to_move(tmp_path):
    with pytest.raises(MoveError, match="No downloaded models"):
        move_models(tmp_path / "empty", tmp_path / "dst")
