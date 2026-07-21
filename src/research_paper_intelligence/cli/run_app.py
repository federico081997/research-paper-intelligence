"""Command-line entry point for the Streamlit application."""

import subprocess
import sys
from pathlib import Path

from research_paper_intelligence.config import get_settings

STREAMLIT_APP_PATH = app_path = (
    Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
)


def main() -> None:
    """Script to run the Streamlit application."""
    settings = get_settings()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            settings.streamlit_host,
            "--server.port",
            str(settings.streamlit_port),
            "--server.headless",
            str(settings.streamlit_headless).lower(),
            "--server.runOnSave",
            str(settings.streamlit_run_on_save).lower(),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
