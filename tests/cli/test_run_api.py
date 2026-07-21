"""Tests for the API command-line entry point."""

from unittest.mock import patch

from research_paper_intelligence.cli.run_api import main
from research_paper_intelligence.config import Settings


class TestMain:
    """Test the API command-line entry point."""

    def test_runs_uvicorn_with_api_settings(
        self,
        simple_settings: Settings,
    ) -> None:
        """Start Uvicorn using the configured application settings."""
        with (
            patch(
                "research_paper_intelligence.cli.run_api.get_settings",
                return_value=simple_settings,
            ) as get_settings_mock,
            patch(
                "research_paper_intelligence.cli.run_api.uvicorn.run",
            ) as uvicorn_run_mock,
        ):
            main()

        get_settings_mock.assert_called_once_with()

        uvicorn_run_mock.assert_called_once_with(
            app="research_paper_intelligence.api.app:fastapi_app",
            host=simple_settings.api_host,
            port=simple_settings.api_port,
            reload=simple_settings.api_reload,
        )
