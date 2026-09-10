"""Canonical defaults shared with the browser. Edit core/config.json."""
import json
from pathlib import Path

CONFIG = json.loads((Path(__file__).resolve().parent.parent / "core/config.json").read_text())

coloring = CONFIG["coloring"]
if coloring["source"] not in ("substrate", "neural"):
    raise ValueError("coloring.source must be substrate or neural")
if len(coloring["channels"]) != 3 or any(type(c) is not int or c < 0 for c in coloring["channels"]):
    raise ValueError("coloring.channels must contain three nonnegative channel indices")

# Flat runtime settings are derived from the grouped defaults, never stored twice.
CHEMISTRY_SETTINGS = {
    key: CONFIG["chemistry"][key]
    for key in (
        'baseResolution',
        'chemicalCommunicationArchitecture',
        'decay',
        'depositRate',
        'normalizeDepositsByLocalDensity',
        'maxEnvWrite',
        'chemicalValueInputMultiplier',
        'chemicalGradientInputScale',
        'communicationSpeed',
        'neuralUpdatesPerMacro',
        'initialConditionChannel',
    )
}
DEFAULT_RUN_SETTINGS = {
    **CONFIG["run"],
    **CHEMISTRY_SETTINGS,
    "channels": len(CONFIG["chemistry"]["channels"]),
}
