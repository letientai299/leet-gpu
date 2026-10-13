import json
from pathlib import Path

import json5

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

notebook_settings = (
    root
    / "lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings"
)
notebook_settings.parent.mkdir(parents=True, exist_ok=True)
notebook_config = (
    json5.loads(notebook_settings.read_text()) if notebook_settings.exists() else {}
)
notebook_config["autoRenderMarkdownCells"] = True
notebook_settings.write_text(json.dumps(notebook_config, indent=4) + "\n")
