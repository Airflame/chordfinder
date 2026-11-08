"""Command line: python -m chordfinder.cli song.mp3 [more files or folders]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import audio_io, models, rhythm
from .analyzer import Callbacks, Pipeline, Settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chordfinder",
                                description="Find the BPM, key, tuning and chords of songs with local models.")
    p.add_argument("paths", nargs="+", type=Path, help="audio files or folders")
    p.add_argument("--chord-model", default=models.DEFAULTS[models.CHORDS],
                   choices=[m.id for m in models.for_task(models.CHORDS)])
    p.add_argument("--beat-model", default=models.DEFAULTS[models.BEATS],
                   choices=[m.id for m in models.for_task(models.BEATS)])
    p.add_argument("--key-model", default=models.DEFAULTS[models.KEY],
                   choices=[m.id for m in models.for_task(models.KEY)])
    p.add_argument("--min-chord", type=float, default=0.4, help="shortest chord in seconds")
    p.add_argument("--no-snap", action="store_true", help="don't move chord changes onto beats")
    p.add_argument("--no-chords", action="store_true", help="print only the summary line")
    p.add_argument("--no-functions", action="store_true", help="omit Roman numerals")
    p.add_argument("--chart", action="store_true", help="also write <song>.chords.txt")
    p.add_argument("--lab", action="store_true", help="print the chords in MIREX .lab format instead")
    p.add_argument("--no-cache", action="store_true", help="always analyse again")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--offline", action="store_true", help="never download models")
    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # ♭, ≈, ø etc. on legacy console code pages
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    s = Settings(chord_model=a.chord_model, beat_model=a.beat_model, key_model=a.key_model,
                 min_chord_seconds=a.min_chord, snap_to_beats=not a.no_snap, use_cache=not a.no_cache,
                 save_chart=a.chart, device="cpu" if a.cpu else "cuda", allow_download=not a.offline)

    def progress(f: float, stage: str) -> None:
        print(f"\r{stage:<16} {f:4.0%}", end="", file=sys.stderr, flush=True)

    files = audio_io.collect(a.paths)
    if not files:
        print("No audio files found.", file=sys.stderr)
        return 1
    pipeline = Pipeline()
    callbacks = Callbacks(log=lambda m: print(f"\r{m:<60}", file=sys.stderr), progress=progress)
    failed = 0
    for path in files:
        try:
            r = pipeline.analyze(path, s, callbacks)
        except Exception as e:
            failed += 1
            print(f"\nFailed: {path}: {e}", file=sys.stderr)
            continue
        print("\r", end="", file=sys.stderr)
        if a.lab:
            print(r.lab(), end="")
            continue
        alternatives = ", ".join(f"{k.name} {p:.0%}" for k, p in r.key_candidates(3)[1:])
        print(f"{path.name}\n  BPM:    {r.tempo.text()}   meter {r.meter_text}\n"
              f"  Key:    {r.key.name} · Camelot {r.key.camelot} · {r.key_confidence:.0%} "
              f"(then {alternatives})\n  Tuning: {rhythm.describe_tuning(r.tuning_cents)}")
        if not a.no_chords:
            print()
            print("\n".join("  " + line for line in r.chart(with_functions=not a.no_functions)
                            .splitlines()[3:]))
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
