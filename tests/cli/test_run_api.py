"""Tests for the API command-line entry point."""

from unittest.mock import Mock

import pytest

from research_paper_intelligence.cli import run_api
from research_paper_intelligence.cli.run_api import main
from research_paper_intelligence.config import Settings


@pytest.fixture
def api_settings() -> Mock:
    """Create settings required by the API CLI."""
    settings = Mock(spec=Settings)
    settings.api_host = "127.0.0.1"
    settings.api_port = 8000
    settings.api_reload = False

    return settings


class TestRunApiMain:
    """Tests for the API CLI entry point."""

    def test_runs_uvicorn_with_configured_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_settings: Mock,
    ) -> None:
        """Run Uvicorn using the configured API settings."""
        get_settings_mock = Mock(return_value=api_settings)
        uvicorn_run_mock = Mock()

        monkeypatch.setattr(
            run_api,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            run_api.uvicorn,
            "run",
            uvicorn_run_mock,
        )

        main()

        get_settings_mock.assert_called_once_with()
        uvicorn_run_mock.assert_called_once_with(
            app=("research_paper_intelligence.api.app:fastapi_app"),
            host="127.0.0.1",
            port=8000,
            reload=False,
        )

    @pytest.mark.parametrize(
        ("host", "port", "reload"),
        [
            ("127.0.0.1", 8000, False),
            ("0.0.0.0", 8080, False),
            ("localhost", 9000, True),
        ],
    )
    def test_passes_different_api_configurations_to_uvicorn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_settings: Mock,
        host: str,
        port: int,
        reload: bool,
    ) -> None:
        """Pass different valid API configurations to Uvicorn."""
        api_settings.api_host = host
        api_settings.api_port = port
        api_settings.api_reload = reload

        uvicorn_run_mock = Mock()

        monkeypatch.setattr(
            run_api,
            "get_settings",
            Mock(return_value=api_settings),
        )
        monkeypatch.setattr(
            run_api.uvicorn,
            "run",
            uvicorn_run_mock,
        )

        main()

        uvicorn_run_mock.assert_called_once_with(
            app=("research_paper_intelligence.api.app:fastapi_app"),
            host=host,
            port=port,
            reload=reload,
        )

    def test_uses_import_string_for_fastapi_application(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_settings: Mock,
    ) -> None:
        """Pass the FastAPI application as an import string."""
        uvicorn_run_mock = Mock()

        monkeypatch.setattr(
            run_api,
            "get_settings",
            Mock(return_value=api_settings),
        )
        monkeypatch.setattr(
            run_api.uvicorn,
            "run",
            uvicorn_run_mock,
        )

        main()

        app_argument = uvicorn_run_mock.call_args.kwargs["app"]

        assert app_argument == (
            "research_paper_intelligence.api.app:fastapi_app"
        )

    def test_propagates_settings_loading_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Propagate errors raised while loading application settings."""
        uvicorn_run_mock = Mock()

        monkeypatch.setattr(
            run_api,
            "get_settings",
            Mock(side_effect=RuntimeError("Settings could not be loaded.")),
        )
        monkeypatch.setattr(
            run_api.uvicorn,
            "run",
            uvicorn_run_mock,
        )

        with pytest.raises(
            RuntimeError,
            match="Settings could not be loaded",
        ):
            main()

        uvicorn_run_mock.assert_not_called()

    def test_propagates_uvicorn_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_settings: Mock,
    ) -> None:
        """Propagate errors raised while starting the Uvicorn server."""
        monkeypatch.setattr(
            run_api,
            "get_settings",
            Mock(return_value=api_settings),
        )
        monkeypatch.setattr(
            run_api.uvicorn,
            "run",
            Mock(side_effect=RuntimeError("Uvicorn failed to start.")),
        )

        with pytest.raises(
            RuntimeError,
            match="Uvicorn failed to start",
        ):
            main()
