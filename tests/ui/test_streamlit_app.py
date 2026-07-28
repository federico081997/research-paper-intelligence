"""Tests for the main Streamlit application entry point."""

import importlib
import sys
from collections.abc import Generator
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import Mock, call

import pytest
import streamlit as st

STREAMLIT_APP_MODULE = "research_paper_intelligence.ui.streamlit_app"
NAVIGATION_MODULE = "research_paper_intelligence.ui.navigation"
STYLES_MODULE = "research_paper_intelligence.ui.styles"


@dataclass(frozen=True)
class StreamlitAppContext:
    """Objects used while importing the Streamlit application."""

    module: ModuleType
    pages: dict[str, list[Mock]]
    set_page_config: Mock
    apply_app_styles: Mock
    navigation: Mock
    selected_page: Mock
    workflow: Mock


@pytest.fixture
def streamlit_app(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[StreamlitAppContext, None, None]:
    """Import the application with its dependencies mocked."""
    home_page = Mock(name="home_page")
    search_page = Mock(name="search_page")
    assistant_page = Mock(name="assistant_page")

    pages = {
        "Discover": [
            home_page,
            search_page,
            assistant_page,
        ]
    }

    selected_page = Mock(name="selected_page")
    set_page_config_mock = Mock()
    apply_app_styles_mock = Mock()
    navigation_mock = Mock(return_value=selected_page)

    navigation_module = ModuleType(NAVIGATION_MODULE)
    navigation_module.PAGES = pages

    styles_module = ModuleType(STYLES_MODULE)
    styles_module.apply_app_styles = apply_app_styles_mock

    monkeypatch.setitem(
        sys.modules,
        NAVIGATION_MODULE,
        navigation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        STYLES_MODULE,
        styles_module,
    )

    monkeypatch.setattr(
        st,
        "set_page_config",
        set_page_config_mock,
    )
    monkeypatch.setattr(
        st,
        "navigation",
        navigation_mock,
    )

    workflow = Mock()
    workflow.attach_mock(
        set_page_config_mock,
        "set_page_config",
    )
    workflow.attach_mock(
        apply_app_styles_mock,
        "apply_app_styles",
    )
    workflow.attach_mock(
        navigation_mock,
        "navigation",
    )
    workflow.attach_mock(
        selected_page.run,
        "run",
    )

    sys.modules.pop(STREAMLIT_APP_MODULE, None)

    module = importlib.import_module(STREAMLIT_APP_MODULE)

    yield StreamlitAppContext(
        module=module,
        pages=pages,
        set_page_config=set_page_config_mock,
        apply_app_styles=apply_app_styles_mock,
        navigation=navigation_mock,
        selected_page=selected_page,
        workflow=workflow,
    )

    sys.modules.pop(STREAMLIT_APP_MODULE, None)


class TestStreamlitApp:
    """Tests for the Streamlit application entry point."""

    def test_configures_streamlit_page(
        self,
        streamlit_app: StreamlitAppContext,
    ) -> None:
        """Configure the page title, icon, layout, and sidebar."""
        streamlit_app.set_page_config.assert_called_once_with(
            page_title="Research Paper Intelligence",
            page_icon=":material/science:",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def test_applies_application_styles(
        self,
        streamlit_app: StreamlitAppContext,
    ) -> None:
        """Apply the shared application styling."""
        streamlit_app.apply_app_styles.assert_called_once_with()

    def test_creates_sidebar_navigation(
        self,
        streamlit_app: StreamlitAppContext,
    ) -> None:
        """Create expanded navigation in the sidebar."""
        streamlit_app.navigation.assert_called_once_with(
            streamlit_app.pages,
            position="sidebar",
            expanded=True,
        )

    def test_runs_selected_navigation_page(
        self,
        streamlit_app: StreamlitAppContext,
    ) -> None:
        """Run the page selected through Streamlit navigation."""
        streamlit_app.selected_page.run.assert_called_once_with()

    def test_stores_selected_page_in_module(
        self,
        streamlit_app: StreamlitAppContext,
    ) -> None:
        """Store the navigation result in the selected-page variable."""
        assert (
            streamlit_app.module.selected_page is streamlit_app.selected_page
        )

    def test_initializes_application_in_expected_order(
        self,
        streamlit_app: StreamlitAppContext,
    ) -> None:
        """Configure, style, navigate, and run the application in order."""
        assert streamlit_app.workflow.mock_calls == [
            call.set_page_config(
                page_title="Research Paper Intelligence",
                page_icon=":material/science:",
                layout="wide",
                initial_sidebar_state="expanded",
            ),
            call.apply_app_styles(),
            call.navigation(
                streamlit_app.pages,
                position="sidebar",
                expanded=True,
            ),
            call.run(),
        ]
