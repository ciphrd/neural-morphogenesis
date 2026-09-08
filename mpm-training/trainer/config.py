"""Canonical defaults shared with the browser. Edit core/config.json."""
import json
from pathlib import Path

CONFIG = json.loads((Path(__file__).resolve().parent.parent / "core/config.json").read_text())

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
