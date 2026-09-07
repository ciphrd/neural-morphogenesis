"""Serve-only mode restores API state without starting or replacing a run."""
import json
import tempfile
from pathlib import Path

import train_server


def main() -> None:
    original = {
        "SETTINGS_PATH": train_server.SETTINGS_PATH,
        "HISTORY_PATH": train_server.HISTORY_PATH,
        "settings": train_server.settings,
        "latest_generation_message": train_server.latest_generation_message,
        "target": train_server.target,
    }
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_server.SETTINGS_PATH = root / "settings.json"
            train_server.HISTORY_PATH = root / "history.jsonl"
            train_server.settings = None
            train_server.latest_generation_message = None
            train_server.target = None

            saved_settings = {"target": "circle", "rasterResolution": 32}
            train_server.SETTINGS_PATH.write_text(json.dumps(saved_settings))
            records = [
                {"generation": 3, "weights": [1]},
                {"generation": 4, "weights": [2]},
            ]
            train_server.HISTORY_PATH.write_text("".join(json.dumps(record) + "\n" for record in records))

            train_server._restore_current_run()

            assert train_server.settings == saved_settings
            assert train_server.latest_generation_message == records[-1]
            assert train_server.target is not None

            train_server.SETTINGS_PATH.unlink()
            train_server.settings = None
            train_server.latest_generation_message = None
            train_server.target = None
            train_server._restore_current_run()
            assert train_server.settings is None
            assert train_server.latest_generation_message is None
            assert train_server.target is None
    finally:
        for name, value in original.items():
            setattr(train_server, name, value)

    parsed = train_server.parser.parse_args(["--serve-only"])
    assert parsed.serve_only is True
    print("[PASS] serve-only restores saved state and tolerates an archive-only server")


if __name__ == "__main__":
    main()
