"""Tests for the global Streamlit application styles."""

from unittest.mock import Mock

import pytest

from research_paper_intelligence.ui import styles
from research_paper_intelligence.ui.styles import apply_app_styles


class TestApplyAppStyles:
    """Tests for the apply_app_styles function."""

    def test_injects_css_using_streamlit_html(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inject the application stylesheet through Streamlit."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        html_mock.assert_called_once()

        stylesheet = html_mock.call_args.args[0]

        assert "<style>" in stylesheet
        assert "</style>" in stylesheet

    @pytest.mark.parametrize(
        "selector",
        [
            'div[class*="st-key-feature-card-"]',
            'div[class*="st-key-result-details-card-"]',
            'div[class*="st-key-result-card-"]',
        ],
    )
    def test_styles_application_cards(
        self,
        monkeypatch: pytest.MonkeyPatch,
        selector: str,
    ) -> None:
        """Include shared styling for application cards."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert selector in stylesheet
        assert "border-radius: 0.9rem;" in stylesheet
        assert "transform: translateY(-3px);" in stylesheet
        assert "border-color: #60a5fa;" in stylesheet

    def test_styles_search_submit_button(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include interaction styles for the search button."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert 'div[class*="st-key-search-submit-button"] button' in stylesheet
        assert (
            'div[class*="st-key-search-submit-button"] button:active'
            in stylesheet
        )
        assert "transform: translateY(1px) scale(0.97);" in stylesheet
        assert "filter: brightness(0.96);" in stylesheet
        assert "@media (prefers-reduced-motion: reduce)" in stylesheet

    def test_styles_search_result_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include layout styles for the search-result list."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert ".st-key-result-list-body" in stylesheet
        assert "padding: 0.75rem 0.85rem 1rem 0.85rem;" in stylesheet
        assert (
            ".st-key-result-list-body "
            'div[class*="st-key-result-card-"]' in stylesheet
        )
        assert "margin-bottom: 0.8rem;" in stylesheet

    def test_styles_clickable_result_titles(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include styles for selectable search-result titles."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert 'div[class*="st-key-result-title-"] button' in stylesheet
        assert (
            'div[class*="st-key-selected-result-title-"] button' in stylesheet
        )
        assert "justify-content: flex-start;" in stylesheet
        assert "white-space: normal;" in stylesheet
        assert "border-left: 4px solid #2563eb;" in stylesheet
        assert "font-weight: 700;" in stylesheet

    def test_styles_selected_result_title(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Highlight the currently selected search result."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert (
            'div[class*="st-key-selected-result-title-"] button {'
            in stylesheet
        )
        assert "padding-left: 0.75rem;" in stylesheet
        assert "border-left: 4px solid #2563eb;" in stylesheet
        assert (
            'div[class*="st-key-selected-result-title-"] button p'
            in stylesheet
        )
        assert "color: #2563eb;" in stylesheet
        assert "font-weight: 700;" in stylesheet

    def test_styles_assistant_chat_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include styles for the assistant conversation container."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert ".st-key-assistant-chat-body" in stylesheet
        assert (
            ".st-key-assistant-chat-body "
            'div[data-testid="stChatMessage"]' in stylesheet
        )
        assert (
            ".st-key-assistant-chat-body "
            'div[data-testid="stChatMessageContent"] p' in stylesheet
        )
        assert "background: rgba(255, 255, 255, 0.86);" in stylesheet
        assert "line-height: 1.65;" in stylesheet

    def test_styles_assistant_empty_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include styles for the assistant's initial empty state."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert ".assistant-empty-state" in stylesheet
        assert ".assistant-empty-state h3" in stylesheet
        assert ".assistant-empty-state p" in stylesheet
        assert "text-align: center;" in stylesheet
        assert "max-width: 650px;" in stylesheet

    def test_styles_assistant_chat_input(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include interaction styles for the assistant input button."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        assert 'div[class*="st-key-assistant-chat-input"] button' in stylesheet
        assert (
            'div[class*="st-key-assistant-chat-input"] button:active'
            in stylesheet
        )
        assert "transform: scale(0.94);" in stylesheet
        assert "filter: brightness(0.95);" in stylesheet

    def test_styles_new_conversation_button(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Include normal, hover, and active button styles."""
        html_mock = Mock()

        monkeypatch.setattr(
            styles.st,
            "html",
            html_mock,
        )

        apply_app_styles()

        stylesheet = html_mock.call_args.args[0]

        button_selector = (
            'div[class*="st-key-assistant-new-conversation-button"] button'
        )

        assert button_selector in stylesheet
        assert f"{button_selector}:hover" in stylesheet
        assert f"{button_selector}:active" in stylesheet
        assert "font-weight: 600;" in stylesheet
        assert "transform: translateY(-2px);" in stylesheet
        assert "transform: translateY(0) scale(0.98);" in stylesheet
