"""Tests for the language-model construction utilities."""

from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel, SecretStr

from research_paper_intelligence.assistant import llm
from research_paper_intelligence.assistant.llm import (
    create_chat_model,
    create_structured_model,
    require_google_api_key,
)
from research_paper_intelligence.config import Settings


class ExampleStructuredOutput(BaseModel):
    """Example schema used to test structured model creation."""

    answer: str


@pytest.fixture
def llm_settings() -> Mock:
    """Create settings containing language-model configuration."""
    settings = Mock(spec=Settings)
    settings.google_api_key = SecretStr("test-google-api-key")
    settings.model_name = "gemini-2.0-flash"

    return settings


class TestRequireGoogleApiKey:
    """Tests for the require_google_api_key function."""

    def test_returns_configured_api_key(
        self,
        llm_settings: Mock,
    ) -> None:
        """Return the Google API key stored in the settings."""
        result = require_google_api_key(llm_settings)

        assert result is llm_settings.google_api_key
        assert result.get_secret_value() == "test-google-api-key"

    def test_raises_runtime_error_when_api_key_is_missing(
        self,
        llm_settings: Mock,
    ) -> None:
        """Raise a clear error when no Google API key is configured."""
        llm_settings.google_api_key = None

        with pytest.raises(
            RuntimeError,
            match=(
                "GOOGLE_API_KEY is required to initialise the language model"
            ),
        ):
            require_google_api_key(llm_settings)


class TestCreateChatModel:
    """Tests for the create_chat_model function."""

    def test_creates_google_chat_model_from_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Mock,
    ) -> None:
        """Create the Google chat model using configured model values."""
        chat_model = Mock()
        chat_model_constructor = Mock(return_value=chat_model)

        monkeypatch.setattr(
            llm,
            "ChatGoogleGenerativeAI",
            chat_model_constructor,
        )

        result = create_chat_model(llm_settings)

        assert result is chat_model
        chat_model_constructor.assert_called_once_with(
            model="gemini-2.0-flash",
            api_key=llm_settings.google_api_key,
        )

    def test_requires_api_key_before_creating_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Mock,
    ) -> None:
        """Reject model creation when the API key is missing."""
        llm_settings.google_api_key = None
        chat_model_constructor = Mock()

        monkeypatch.setattr(
            llm,
            "ChatGoogleGenerativeAI",
            chat_model_constructor,
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "GOOGLE_API_KEY is required to initialise the language model"
            ),
        ):
            create_chat_model(llm_settings)

        chat_model_constructor.assert_not_called()

    def test_uses_api_key_returned_by_validation_function(
        self,
        monkeypatch: pytest.MonkeyPatch,
        llm_settings: Mock,
    ) -> None:
        """Pass the validated API key to the Google model constructor."""
        validated_api_key = SecretStr("validated-api-key")
        require_api_key_mock = Mock(return_value=validated_api_key)
        chat_model_constructor = Mock()

        monkeypatch.setattr(
            llm,
            "require_google_api_key",
            require_api_key_mock,
        )
        monkeypatch.setattr(
            llm,
            "ChatGoogleGenerativeAI",
            chat_model_constructor,
        )

        create_chat_model(llm_settings)

        require_api_key_mock.assert_called_once_with(llm_settings)
        chat_model_constructor.assert_called_once_with(
            model=llm_settings.model_name,
            api_key=validated_api_key,
        )


class TestCreateStructuredModel:
    """Tests for the create_structured_model function."""

    def test_binds_chat_model_to_output_schema(self) -> None:
        """Bind the chat model to the supplied Pydantic schema."""
        structured_model = Mock(spec=Runnable)
        chat_model = Mock(spec=BaseChatModel)
        chat_model.with_structured_output.return_value = structured_model

        result = create_structured_model(
            chat_model=chat_model,
            output_schema=ExampleStructuredOutput,
        )

        assert result is structured_model
        chat_model.with_structured_output.assert_called_once_with(
            ExampleStructuredOutput
        )

    def test_returns_exact_structured_model_from_chat_model(self) -> None:
        """Return the runnable produced by with_structured_output."""
        first_structured_model = Mock(spec=Runnable)
        second_structured_model = Mock(spec=Runnable)
        chat_model = Mock(spec=BaseChatModel)

        chat_model.with_structured_output.side_effect = [
            first_structured_model,
            second_structured_model,
        ]

        first_result = create_structured_model(
            chat_model=chat_model,
            output_schema=ExampleStructuredOutput,
        )
        second_result = create_structured_model(
            chat_model=chat_model,
            output_schema=ExampleStructuredOutput,
        )

        assert first_result is first_structured_model
        assert second_result is second_structured_model
