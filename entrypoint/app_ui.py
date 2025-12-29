"""Programmatic entrypoint for running the Dash UI application."""

import os
import sys
from pathlib import Path

# Add src and app_ui directories to path before imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "app_ui"))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)

from app_ui.app import app  # noqa: E402

if __name__ == "__main__":
    # Use debug=False in production/Docker, debug=True for local development
    debug_mode = os.getenv("DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode, use_reloader=False, host="0.0.0.0", port=8050)
