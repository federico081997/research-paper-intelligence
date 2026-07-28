"""Tests for the Streamlit navigation configuration."""

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from unittest.mock import Mock, call

import pytest
import streamlit as st

NAVIGATION_MODULE = "research_paper_intelligence.ui.navigation"


@pytest.fixture
def navigation_pages() -> tuple[Mock, Mock, Mock]:
    """Create mocked Streamlit navigation pages."""
    return (
        Mock(name="home_page"),
        Mock(name="search_page"),
        Mock(name="assistant_page"),
    )


@pytest.fixture
def page_constructor(
    monkeypatch: pytest.MonkeyPatch,
    navigation_pages: tuple[Mock, Mock, Mock],
) -> Mock:
    """Replace the Streamlit Page constructor with a mock."""
    constructor = Mock(side_effect=navigation_pages)

    monkeypatch.setattr(
        st,
        "Page",
        constructor,
    )

    return constructor


@pytest.fixture
def navigation_module(
    page_constructor: Mock,
) -> Iterator[ModuleType]:
    """Import the navigation module using the mocked Page constructor."""
    sys.modules.pop(NAVIGATION_MODULE, None)

    module = importlib.import_module(NAVIGATION_MODULE)

    yield module

    sys.modules.pop(NAVIGATION_MODULE, None)


class TestNavigationPages:
    """Tests for the Streamlit navigation-page definitions."""

    def test_creates_expected_navigation_pages(
        self,
        navigation_module: ModuleType,
        page_constructor: Mock,
    ) -> None:
        """Create the home, search, and assistant navigation pages."""
        assert page_constructor.call_args_list == [
            call(
                "pages/home.py",
                title="Home",
                icon=":material/home:",
                url_path="home",
                default=True,
            ),
            call(
                "pages/search.py",
                title="Paper Search",
                icon=":material/search:",
                url_path="search",
            ),
            call(
                "pages/assistant.py",
                title="Research Assistant",
                icon=":material/smart_toy:",
                url_path="assistant",
            ),
        ]

    def test_assigns_constructed_pages_to_module_constants(
        self,
        navigation_module: ModuleType,
        navigation_pages: tuple[Mock, Mock, Mock],
    ) -> None:
        """Assign each constructed page to the expected module constant."""
        home_page, search_page, assistant_page = navigation_pages

        assert navigation_module.HOME_PAGE is home_page
        assert navigation_module.SEARCH_PAGE is search_page
        assert navigation_module.ASSISTANT_PAGE is assistant_page

    def test_groups_pages_under_discover_section(
        self,
        navigation_module: ModuleType,
        navigation_pages: tuple[Mock, Mock, Mock],
    ) -> None:
        """Group all application pages under the Discover section."""
        home_page, search_page, assistant_page = navigation_pages

        assert navigation_module.PAGES == {
            "Discover": [
                home_page,
                search_page,
                assistant_page,
            ]
        }

    def test_preserves_navigation_page_order(
        self,
        navigation_module: ModuleType,
        navigation_pages: tuple[Mock, Mock, Mock],
    ) -> None:
        """List home, search, and assistant pages in display order."""
        assert navigation_module.PAGES["Discover"] == list(navigation_pages)

    def test_marks_only_home_page_as_default(
        self,
        navigation_module: ModuleType,
        page_constructor: Mock,
    ) -> None:
        """Configure the home page as the only default page."""
        calls = page_constructor.call_args_list

        assert calls[0].kwargs["default"] is True
        assert "default" not in calls[1].kwargs
        assert "default" not in calls[2].kwargs
