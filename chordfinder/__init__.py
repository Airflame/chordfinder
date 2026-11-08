"""Chordfinder: offline BPM, key, tuning and chord analysis with local models."""

import os

__version__ = "0.1.0"

# Models are stored locally; never phone home.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
