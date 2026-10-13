import json
from pathlib import Path

root = Path(__file__).resolve().parent
settings = (
    root / "lab/user-settings/@jupyterlab/plugin-playground/plugin.jupyterlab-settings"
)
settings.parent.mkdir(parents=True, exist_ok=True)
settings.write_text(
    json.dumps(
        {
            "allowCDN": "never",
            "showIconInLauncher": False,
            "plugins": [(root / "autoreload.js").read_text()],
        },
        indent=2,
    )
    + "\n"
)
