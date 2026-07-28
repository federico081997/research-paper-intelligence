"""Tests for the Streamlit application command-line entry point."""

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from research_paper_intelligence.cli import run_app
from research_paper_intelligence.cli.run_app import main
from research_paper_intelligence.config import Settings


@pytest.fixture
def streamlit_settings() -> Mock:
    """Create settings required by the Streamlit CLI."""
    settings = Mock(spec=Settings)
    settings.streamlit_host = "127.0.0.1"
    settings.streamlit_port = 8501
    settings.streamlit_headless = True
    settings.streamlit_run_on_save = False

    return settings


class TestRunAppMain:
    """Tests for the Streamlit CLI entry point."""

    def test_runs_streamlit_with_configured_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        streamlit_settings: Mock,
    ) -> None:
        """Run Streamlit using the configured application settings."""
        app_path = Path("/project/ui/streamlit_app.py")
        get_settings_mock = Mock(return_value=streamlit_settings)
        subprocess_run_mock = Mock()

        monkeypatch.setattr(
            run_app,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            run_app,
            "app_path",
            app_path,
        )
        monkeypatch.setattr(
            run_app.sys,
            "executable",
            "/usr/bin/python",
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            subprocess_run_mock,
        )

        main()

        get_settings_mock.assert_called_once_with()
        subprocess_run_mock.assert_called_once_with(
            [
                "/usr/bin/python",
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                "8501",
                "--server.headless",
                "true",
                "--server.runOnSave",
                "false",
            ],
            check=True,
        )

    @pytest.mark.parametrize(
        (
            "host",
            "port",
            "headless",
            "run_on_save",
            "expected_headless",
            "expected_run_on_save",
        ),
        [
            (
                "127.0.0.1",
                8501,
                True,
                False,
                "true",
                "false",
            ),
            (
                "0.0.0.0",
                9000,
                False,
                True,
                "false",
                "true",
            ),
        ],
    )
    def test_passes_different_configurations_to_streamlit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        streamlit_settings: Mock,
        host: str,
        port: int,
        headless: bool,
        run_on_save: bool,
        expected_headless: str,
        expected_run_on_save: str,
    ) -> None:
        """Pass configured server values to the Streamlit command."""
        streamlit_settings.streamlit_host = host
        streamlit_settings.streamlit_port = port
        streamlit_settings.streamlit_headless = headless
        streamlit_settings.streamlit_run_on_save = run_on_save

        app_path = Path("/project/ui/streamlit_app.py")
        subprocess_run_mock = Mock()

        monkeypatch.setattr(
            run_app,
            "get_settings",
            Mock(return_value=streamlit_settings),
        )
        monkeypatch.setattr(
            run_app,
            "app_path",
            app_path,
        )
        monkeypatch.setattr(
            run_app.sys,
            "executable",
            "/usr/bin/python",
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            subprocess_run_mock,
        )

        main()

        command = subprocess_run_mock.call_args.args[0]

        assert command == [
            "/usr/bin/python",
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            host,
            "--server.port",
            str(port),
            "--server.headless",
            expected_headless,
            "--server.runOnSave",
            expected_run_on_save,
        ]

    def test_uses_current_python_interpreter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        streamlit_settings: Mock,
    ) -> None:
        """Launch Streamlit using the current Python interpreter."""
        subprocess_run_mock = Mock()

        monkeypatch.setattr(
            run_app,
            "get_settings",
            Mock(return_value=streamlit_settings),
        )
        monkeypatch.setattr(
            run_app.sys,
            "executable",
            "/custom/python",
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            subprocess_run_mock,
        )

        main()

        command = subprocess_run_mock.call_args.args[0]

        assert command[0] == "/custom/python"
        assert command[1:4] == [
            "-m",
            "streamlit",
            "run",
        ]

    def test_uses_configured_streamlit_application_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        streamlit_settings: Mock,
    ) -> None:
        """Pass the Streamlit application path to the command."""
        app_path = Path("/custom/path/streamlit_app.py")
        subprocess_run_mock = Mock()

        monkeypatch.setattr(
            run_app,
            "get_settings",
            Mock(return_value=streamlit_settings),
        )
        monkeypatch.setattr(
            run_app,
            "app_path",
            app_path,
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            subprocess_run_mock,
        )

        main()

        command = subprocess_run_mock.call_args.args[0]

        assert command[4] == str(app_path)

    def test_requires_successful_subprocess_execution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        streamlit_settings: Mock,
    ) -> None:
        """Request an exception when the Streamlit process fails."""
        subprocess_run_mock = Mock()

        monkeypatch.setattr(
            run_app,
            "get_settings",
            Mock(return_value=streamlit_settings),
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            subprocess_run_mock,
        )

        main()

        assert subprocess_run_mock.call_args.kwargs["check"] is True

    def test_propagates_settings_loading_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Propagate errors raised while loading application settings."""
        subprocess_run_mock = Mock()

        monkeypatch.setattr(
            run_app,
            "get_settings",
            Mock(side_effect=RuntimeError("Settings could not be loaded.")),
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            subprocess_run_mock,
        )

        with pytest.raises(
            RuntimeError,
            match="Settings could not be loaded",
        ):
            main()

        subprocess_run_mock.assert_not_called()

    def test_propagates_streamlit_process_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        streamlit_settings: Mock,
    ) -> None:
        """Propagate errors raised when the Streamlit process fails."""
        process_error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "-m", "streamlit", "run"],
        )

        monkeypatch.setattr(
            run_app,
            "get_settings",
            Mock(return_value=streamlit_settings),
        )
        monkeypatch.setattr(
            run_app.subprocess,
            "run",
            Mock(side_effect=process_error),
        )

        with pytest.raises(
            subprocess.CalledProcessError,
        ) as exc_info:
            main()

        assert exc_info.value is process_error
