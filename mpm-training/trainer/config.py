"""Canonical defaults shared with the browser. Edit core/config.json."""
import json
from pathlib import Path

CONFIG = json.loads((Path(__file__).resolve().parent.parent / "core/config.json").read_text())
