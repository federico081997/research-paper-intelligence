"""Tests for the research-assistant Streamlit page."""

import importlib
import sys
from collections.abc import Generator
from types import ModuleType
from unittest.mock import MagicMock, Mock, call
from uuid import UUID, uuid4

import pytest
import streamlit as st

from research_paper_intelligence.api.schemas.assistant import (
    AssistantResponse,
)

ASSISTANT_PAGE_MODULE = "research_paper_intelligence.ui.pages.assistant"


@pytest.fixture
def context_manager() -> MagicMock:
    """Create a mocked context manager."""
    context = MagicMock()
    context.__enter__.return_value = context
    context.__exit__.return_value = False

    return context


@pytest.fixture
def assistant_page(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[ModuleType, None, None]:
    """Import the assistant page with its Streamlit calls mocked."""
    session_state: dict[str, object] = {}

    column_one = MagicMock()
    column_two = MagicMock()
    container = MagicMock()
    chat_message = MagicMock()
    spinner = MagicMock()

    monkeypatch.setattr(
        st,
        "session_state",
        session_state,
        raising=False,
    )
    monkeypatch.setattr(st, "title", Mock())
    monkeypatch.setattr(st, "markdown", Mock())
    monkeypatch.setattr(
        st,
        "columns",
        Mock(return_value=[column_one, column_two]),
    )
    monkeypatch.setattr(st, "button", Mock())
    monkeypatch.setattr(
        st,
        "chat_input",
        Mock(return_value=None),
    )
    monkeypatch.setattr(
        st,
        "container",
        Mock(return_value=container),
    )
    monkeypatch.setattr(
        st,
        "chat_message",
        Mock(return_value=chat_message),
    )
    monkeypatch.setattr(
        st,
        "spinner",
        Mock(return_value=spinner),
    )
    monkeypatch.setattr(st, "error", Mock())

    sys.modules.pop(ASSISTANT_PAGE_MODULE, None)

    module = importlib.import_module(ASSISTANT_PAGE_MODULE)

    # Remove state created by the module-level render_assistant_page()
    # invocation so each test starts with an isolated session.
    session_state.clear()

    yield module

    sys.modules.pop(ASSISTANT_PAGE_MODULE, None)


class TestInitializeAssistantState:
    """Tests for assistant session-state initialization."""

    def test_initializes_missing_session_state_values(
        self,
        assistant_page: ModuleType,
    ) -> None:
        """Create all required state values when they are missing."""
        assistant_page.initialize_assistant_state()

        state = assistant_page.st.session_state

        assert set(state) == {
            assistant_page.THREAD_ID_KEY,
            assistant_page.MESSAGES_KEY,
            assistant_page.PENDING_PROMPT_KEY,
            assistant_page.FAILED_PROMPT_KEY,
        }
        assert UUID(
            state[assistant_page.THREAD_ID_KEY],
            version=4,
        )
        assert state[assistant_page.MESSAGES_KEY] == []
        assert state[assistant_page.PENDING_PROMPT_KEY] is None
        assert state[assistant_page.FAILED_PROMPT_KEY] is None

    def test_preserves_existing_session_state_values(
        self,
        assistant_page: ModuleType,
    ) -> None:
        """Leave previously initialized state values unchanged."""
        existing_messages = [
            {
                "role": "user",
                "content": "Existing message",
            }
        ]

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: "existing-thread",
                assistant_page.MESSAGES_KEY: existing_messages,
                assistant_page.PENDING_PROMPT_KEY: "pending prompt",
                assistant_page.FAILED_PROMPT_KEY: "failed prompt",
            }
        )

        assistant_page.initialize_assistant_state()

        state = assistant_page.st.session_state

        assert state[assistant_page.THREAD_ID_KEY] == "existing-thread"
        assert state[assistant_page.MESSAGES_KEY] is existing_messages
        assert state[assistant_page.PENDING_PROMPT_KEY] == "pending prompt"
        assert state[assistant_page.FAILED_PROMPT_KEY] == "failed prompt"


class TestStartNewConversation:
    """Tests for starting a new assistant conversation."""

    def test_resets_conversation_state(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clear messages and create a new conversation thread."""
        new_thread_id = UUID("123e4567-e89b-42d3-a456-426614174000")

        monkeypatch.setattr(
            assistant_page,
            "uuid4",
            Mock(return_value=new_thread_id),
        )

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: "old-thread",
                assistant_page.MESSAGES_KEY: [
                    {
                        "role": "user",
                        "content": "Old message",
                    }
                ],
                assistant_page.PENDING_PROMPT_KEY: "pending prompt",
                assistant_page.FAILED_PROMPT_KEY: "failed prompt",
            }
        )

        assistant_page.start_new_conversation()

        state = assistant_page.st.session_state

        assert state[assistant_page.THREAD_ID_KEY] == str(new_thread_id)
        assert state[assistant_page.MESSAGES_KEY] == []
        assert state[assistant_page.PENDING_PROMPT_KEY] is None
        assert state[assistant_page.FAILED_PROMPT_KEY] is None


class TestQueuePrompt:
    """Tests for queuing suggested prompts."""

    def test_stores_prompt_in_session_state(
        self,
        assistant_page: ModuleType,
    ) -> None:
        """Store the selected prompt for processing on the next rerun."""
        assistant_page.queue_prompt("Find papers about semantic search.")

        assert (
            assistant_page.st.session_state[assistant_page.PENDING_PROMPT_KEY]
            == "Find papers about semantic search."
        )


class TestRenderHero:
    """Tests for rendering the page hero."""

    def test_renders_title_and_description(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the assistant title and introductory description."""
        title_mock = Mock()
        markdown_mock = Mock()

        monkeypatch.setattr(
            assistant_page.st,
            "title",
            title_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            markdown_mock,
        )

        assistant_page.render_hero()

        title_mock.assert_called_once_with("Research Assistant")

        description = markdown_mock.call_args.args[0]

        assert "Ask scientific and technical questions" in description
        assert "evidence-grounded answers" in description
        assert "citations" in description


class TestRenderConversationControls:
    """Tests for rendering conversation controls."""

    def test_renders_new_conversation_button(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the button that starts a new conversation."""
        first_column = MagicMock()
        button_column = MagicMock()
        columns_mock = Mock(return_value=[first_column, button_column])
        button_mock = Mock()

        monkeypatch.setattr(
            assistant_page.st,
            "columns",
            columns_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "button",
            button_mock,
        )

        assistant_page.render_conversation_controls()

        columns_mock.assert_called_once_with(
            [0.8, 0.2],
            vertical_alignment="center",
        )
        button_column.__enter__.assert_called_once_with()
        button_column.__exit__.assert_called_once()

        button_mock.assert_called_once_with(
            "New conversation",
            icon=":material/add_comment:",
            type="secondary",
            width="stretch",
            key="assistant-new-conversation-button",
            on_click=assistant_page.start_new_conversation,
        )


class TestRenderMessage:
    """Tests for rendering individual chat messages."""

    @pytest.mark.parametrize(
        ("role", "expected_avatar"),
        [
            ("user", ":material/person:"),
            ("assistant", ":material/smart_toy:"),
        ],
    )
    def test_renders_message_with_role_avatar(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
        role: str,
        expected_avatar: str,
    ) -> None:
        """Render a chat message with the avatar for its role."""
        chat_message_mock = Mock(return_value=context_manager)
        markdown_mock = Mock()

        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            chat_message_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            markdown_mock,
        )

        message = {
            "role": role,
            "content": "Example message",
        }

        assistant_page.render_message(message)

        chat_message_mock.assert_called_once_with(
            role,
            avatar=expected_avatar,
        )
        markdown_mock.assert_called_once_with("Example message")
        context_manager.__enter__.assert_called_once_with()
        context_manager.__exit__.assert_called_once()


class TestRenderChatHistory:
    """Tests for rendering stored chat history."""

    def test_renders_messages_in_stored_order(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render every stored message in conversation order."""
        messages = [
            {
                "role": "user",
                "content": "First message",
            },
            {
                "role": "assistant",
                "content": "Second message",
            },
            {
                "role": "user",
                "content": "Third message",
            },
        ]
        render_message_mock = Mock()

        assistant_page.st.session_state[assistant_page.MESSAGES_KEY] = messages

        monkeypatch.setattr(
            assistant_page,
            "render_message",
            render_message_mock,
        )

        assistant_page.render_chat_history()

        assert render_message_mock.call_args_list == [
            call(messages[0]),
            call(messages[1]),
            call(messages[2]),
        ]

    def test_renders_nothing_for_empty_history(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render no messages when conversation history is empty."""
        render_message_mock = Mock()

        assistant_page.st.session_state[assistant_page.MESSAGES_KEY] = []

        monkeypatch.setattr(
            assistant_page,
            "render_message",
            render_message_mock,
        )

        assistant_page.render_chat_history()

        render_message_mock.assert_not_called()


class TestRenderEmptyState:
    """Tests for rendering the assistant empty state."""

    def test_renders_empty_state_description(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render the introductory empty-state HTML."""
        markdown_mock = Mock()
        first_column = MagicMock()
        second_column = MagicMock()

        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            markdown_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "columns",
            Mock(return_value=[first_column, second_column]),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "button",
            Mock(),
        )

        assistant_page.render_empty_state()

        html = markdown_mock.call_args.args[0]

        assert "What would you like to investigate?" in html
        assert "Ask a scientific question" in html
        assert markdown_mock.call_args.kwargs["unsafe_allow_html"] is True

    def test_creates_two_prompt_columns(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Arrange suggested prompts in two columns."""
        columns_mock = Mock(return_value=[MagicMock(), MagicMock()])

        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "columns",
            columns_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "button",
            Mock(),
        )

        assistant_page.render_empty_state()

        columns_mock.assert_called_once_with(2)

    def test_renders_all_suggested_prompt_buttons(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render one button for every configured suggested prompt."""
        prompt_columns = [
            MagicMock(),
            MagicMock(),
        ]
        button_mock = Mock()

        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "columns",
            Mock(return_value=prompt_columns),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "button",
            button_mock,
        )

        assistant_page.render_empty_state()

        expected_calls = []

        for index, (label, prompt) in enumerate(
            assistant_page.SUGGESTED_PROMPTS
        ):
            expected_calls.append(
                call(
                    label,
                    icon=":material/arrow_forward:",
                    help=prompt,
                    width="stretch",
                    key=f"assistant-suggested-prompt-{index}",
                    on_click=assistant_page.queue_prompt,
                    args=(prompt,),
                )
            )

        assert button_mock.call_args_list == expected_calls


class TestProcessPrompt:
    """Tests for processing submitted assistant prompts."""

    def test_ignores_blank_prompt(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Return without processing an empty or whitespace-only prompt."""
        messages: list[dict[str, str]] = []

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: str(uuid4()),
                assistant_page.MESSAGES_KEY: messages,
                assistant_page.FAILED_PROMPT_KEY: None,
            }
        )

        chat_mock = Mock()
        render_message_mock = Mock()

        monkeypatch.setattr(
            assistant_page,
            "chat",
            chat_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            render_message_mock,
        )

        assistant_page.process_prompt("   ")

        assert messages == []
        chat_mock.assert_not_called()
        render_message_mock.assert_not_called()

    def test_sends_cleaned_prompt_to_assistant(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Strip whitespace before sending the prompt to the backend."""
        current_thread_id = str(uuid4())
        response_thread_id = uuid4()

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: current_thread_id,
                assistant_page.MESSAGES_KEY: [],
                assistant_page.FAILED_PROMPT_KEY: ("previous failed prompt"),
            }
        )

        response = AssistantResponse(
            response="Grounded assistant answer.",
            thread_id=response_thread_id,
        )
        chat_mock = Mock(return_value=response)

        chat_message_context = MagicMock()
        spinner_context = MagicMock()

        monkeypatch.setattr(
            assistant_page,
            "chat",
            chat_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            Mock(return_value=chat_message_context),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "spinner",
            Mock(return_value=spinner_context),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            Mock(),
        )

        assistant_page.process_prompt("  Explain finite volume methods.  ")

        chat_mock.assert_called_once_with(
            user_query="Explain finite volume methods.",
            thread_id=current_thread_id,
        )

    def test_stores_user_and_assistant_messages(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Store the submitted prompt and successful assistant answer."""
        current_thread_id = str(uuid4())
        response_thread_id = uuid4()

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: current_thread_id,
                assistant_page.MESSAGES_KEY: [],
                assistant_page.FAILED_PROMPT_KEY: ("previous failed prompt"),
            }
        )

        response = AssistantResponse(
            response="Grounded assistant answer.",
            thread_id=response_thread_id,
        )
        render_message_mock = Mock()
        markdown_mock = Mock()

        monkeypatch.setattr(
            assistant_page,
            "chat",
            Mock(return_value=response),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            render_message_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "spinner",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            markdown_mock,
        )

        assistant_page.process_prompt("Explain finite volume methods.")

        expected_user_message = {
            "role": "user",
            "content": "Explain finite volume methods.",
        }
        expected_assistant_message = {
            "role": "assistant",
            "content": "Grounded assistant answer.",
        }

        assert assistant_page.st.session_state[
            assistant_page.MESSAGES_KEY
        ] == [
            expected_user_message,
            expected_assistant_message,
        ]
        assert (
            assistant_page.st.session_state[assistant_page.FAILED_PROMPT_KEY]
            is None
        )

        render_message_mock.assert_called_once_with(expected_user_message)
        markdown_mock.assert_called_once_with("Grounded assistant answer.")

    def test_updates_thread_id_from_response(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Store the thread ID returned by the assistant backend."""
        response_thread_id = uuid4()

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: str(uuid4()),
                assistant_page.MESSAGES_KEY: [],
                assistant_page.FAILED_PROMPT_KEY: None,
            }
        )

        monkeypatch.setattr(
            assistant_page,
            "chat",
            Mock(
                return_value=AssistantResponse(
                    response="Assistant answer.",
                    thread_id=response_thread_id,
                )
            ),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "spinner",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            Mock(),
        )

        assistant_page.process_prompt("Example prompt")

        assert assistant_page.st.session_state[
            assistant_page.THREAD_ID_KEY
        ] == str(response_thread_id)

    def test_renders_assistant_spinner(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Render an assistant message container and progress spinner."""
        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: str(uuid4()),
                assistant_page.MESSAGES_KEY: [],
                assistant_page.FAILED_PROMPT_KEY: None,
            }
        )

        chat_message_context = MagicMock()
        spinner_context = MagicMock()
        chat_message_mock = Mock(return_value=chat_message_context)
        spinner_mock = Mock(return_value=spinner_context)

        monkeypatch.setattr(
            assistant_page,
            "chat",
            Mock(
                return_value=AssistantResponse(
                    response="Assistant answer.",
                    thread_id=uuid4(),
                )
            ),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            chat_message_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "spinner",
            spinner_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "markdown",
            Mock(),
        )

        assistant_page.process_prompt("Example prompt")

        chat_message_mock.assert_called_once_with(
            "assistant",
            avatar=":material/smart_toy:",
        )
        spinner_mock.assert_called_once_with(
            "Searching the literature and preparing the answer...",
            show_time=True,
        )
        chat_message_context.__enter__.assert_called_once_with()
        chat_message_context.__exit__.assert_called_once()
        spinner_context.__enter__.assert_called_once_with()
        spinner_context.__exit__.assert_called_once()

    def test_removes_unanswered_message_when_request_fails(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Remove the temporary user message after a backend failure."""
        existing_message = {
            "role": "assistant",
            "content": "Existing answer",
        }
        messages = [existing_message]
        current_thread_id = str(uuid4())

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: current_thread_id,
                assistant_page.MESSAGES_KEY: messages,
                assistant_page.FAILED_PROMPT_KEY: None,
            }
        )

        logger_mock = Mock()
        error_mock = Mock()

        monkeypatch.setattr(
            assistant_page,
            "chat",
            Mock(side_effect=RuntimeError("Assistant unavailable.")),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "LOGGER",
            logger_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "spinner",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "error",
            error_mock,
        )

        assistant_page.process_prompt("  Failed research question.  ")

        assert messages == [existing_message]
        assert (
            assistant_page.st.session_state[assistant_page.FAILED_PROMPT_KEY]
            == "Failed research question."
        )
        assert (
            assistant_page.st.session_state[assistant_page.THREAD_ID_KEY]
            == current_thread_id
        )

        logger_mock.exception.assert_called_once_with(
            "Assistant request failed for thread %s.",
            current_thread_id,
        )
        error_mock.assert_called_once_with(
            "The message request failed. Please try again.",
            icon=":material/error:",
        )

    def test_does_not_store_assistant_message_when_request_fails(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Store no new messages when the backend request fails."""
        messages: list[dict[str, str]] = []

        assistant_page.st.session_state.update(
            {
                assistant_page.THREAD_ID_KEY: str(uuid4()),
                assistant_page.MESSAGES_KEY: messages,
                assistant_page.FAILED_PROMPT_KEY: None,
            }
        )

        monkeypatch.setattr(
            assistant_page,
            "chat",
            Mock(side_effect=RuntimeError("Failure")),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_message",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_message",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "spinner",
            Mock(return_value=MagicMock()),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "error",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page.LOGGER,
            "exception",
            Mock(),
        )

        assistant_page.process_prompt("Example prompt")

        assert messages == []


class TestRenderAssistantPage:
    """Tests for rendering the complete assistant page."""

    def test_renders_page_sections_and_chat_input(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Render the page hero, controls, input, and chat container."""
        hero_mock = Mock()
        controls_mock = Mock()
        empty_state_mock = Mock()
        container_mock = Mock(return_value=context_manager)
        chat_input_mock = Mock(return_value=None)

        monkeypatch.setattr(
            assistant_page,
            "render_hero",
            hero_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "render_conversation_controls",
            controls_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "render_empty_state",
            empty_state_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "chat_input",
            chat_input_mock,
        )
        monkeypatch.setattr(
            assistant_page.st,
            "container",
            container_mock,
        )

        assistant_page.render_assistant_page()

        hero_mock.assert_called_once_with()
        controls_mock.assert_called_once_with()

        chat_input_mock.assert_called_once_with(
            "Ask a scientific question or request relevant papers...",
            key="assistant-chat-input",
            max_chars=2_000,
            accept_audio=False,
            accept_file=False,
        )
        container_mock.assert_called_once_with(
            height=400,
            border=True,
            key="assistant-chat-body",
        )
        empty_state_mock.assert_called_once_with()

    def test_processes_submitted_chat_prompt(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Process the prompt returned by the chat input."""
        process_prompt_mock = Mock()
        empty_state_mock = Mock()

        monkeypatch.setattr(
            assistant_page.st,
            "chat_input",
            Mock(return_value="Submitted prompt"),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "container",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_conversation_controls",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_empty_state",
            empty_state_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "process_prompt",
            process_prompt_mock,
        )

        assistant_page.render_assistant_page()

        process_prompt_mock.assert_called_once_with("Submitted prompt")
        empty_state_mock.assert_not_called()

    def test_processes_pending_suggested_prompt(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Process a suggested prompt stored in session state."""
        process_prompt_mock = Mock()

        assistant_page.st.session_state[assistant_page.PENDING_PROMPT_KEY] = (
            "Suggested prompt"
        )

        monkeypatch.setattr(
            assistant_page.st,
            "chat_input",
            Mock(return_value=None),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "container",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_conversation_controls",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_empty_state",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "process_prompt",
            process_prompt_mock,
        )

        assistant_page.render_assistant_page()

        process_prompt_mock.assert_called_once_with("Suggested prompt")
        assert (
            assistant_page.PENDING_PROMPT_KEY
            not in assistant_page.st.session_state
        )

    def test_prioritizes_submitted_prompt_over_pending_prompt(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Prefer typed input when both prompt sources are available."""
        process_prompt_mock = Mock()

        assistant_page.st.session_state[assistant_page.PENDING_PROMPT_KEY] = (
            "Pending prompt"
        )

        monkeypatch.setattr(
            assistant_page.st,
            "chat_input",
            Mock(return_value="Typed prompt"),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "container",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_conversation_controls",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "process_prompt",
            process_prompt_mock,
        )

        assistant_page.render_assistant_page()

        process_prompt_mock.assert_called_once_with("Typed prompt")
        assert (
            assistant_page.PENDING_PROMPT_KEY
            not in assistant_page.st.session_state
        )

    def test_renders_chat_history_when_messages_exist(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Render stored history instead of the empty state."""
        history_mock = Mock()
        empty_state_mock = Mock()
        process_prompt_mock = Mock()

        assistant_page.st.session_state.update(
            {
                assistant_page.MESSAGES_KEY: [
                    {
                        "role": "user",
                        "content": "Stored message",
                    }
                ],
                assistant_page.PENDING_PROMPT_KEY: None,
            }
        )

        monkeypatch.setattr(
            assistant_page.st,
            "chat_input",
            Mock(return_value=None),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "container",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_conversation_controls",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_chat_history",
            history_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "render_empty_state",
            empty_state_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "process_prompt",
            process_prompt_mock,
        )

        assistant_page.render_assistant_page()

        history_mock.assert_called_once_with()
        empty_state_mock.assert_not_called()
        process_prompt_mock.assert_not_called()

    def test_renders_empty_state_without_messages_or_prompt(
        self,
        assistant_page: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        context_manager: MagicMock,
    ) -> None:
        """Render the empty state before the first submitted prompt."""
        history_mock = Mock()
        empty_state_mock = Mock()
        process_prompt_mock = Mock()

        monkeypatch.setattr(
            assistant_page.st,
            "chat_input",
            Mock(return_value=None),
        )
        monkeypatch.setattr(
            assistant_page.st,
            "container",
            Mock(return_value=context_manager),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_hero",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_conversation_controls",
            Mock(),
        )
        monkeypatch.setattr(
            assistant_page,
            "render_chat_history",
            history_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "render_empty_state",
            empty_state_mock,
        )
        monkeypatch.setattr(
            assistant_page,
            "process_prompt",
            process_prompt_mock,
        )

        assistant_page.render_assistant_page()

        history_mock.assert_not_called()
        empty_state_mock.assert_called_once_with()
        process_prompt_mock.assert_not_called()
