"""Page for the research paper assistant."""

import logging
from typing import Literal, TypedDict
from uuid import uuid4

import streamlit as st

from research_paper_intelligence.api.schemas.assistant import AssistantResponse
from research_paper_intelligence.ui.api_clients.assistant_client import chat

THREAD_ID_KEY = "assistant_thread_id"
MESSAGES_KEY = "assistant_messages"
PENDING_PROMPT_KEY = "assistant_pending_prompt"
FAILED_PROMPT_KEY = "assistant_failed_prompt"

LOGGER = logging.getLogger(__name__)

SUGGESTED_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "Find research papers",
        "Find recent research papers about retrieval-augmented generation.",
    ),
    (
        "Compare methods",
        "Compare finite-volume and finite-element methods for solid "
        "mechanics.",
    ),
    (
        "Explain a concept",
        "Explain how semantic search using embeddings and FAISS works.",
    ),
    (
        "Review evidence",
        "How should retrieval-augmented generation systems be evaluated?",
    ),
)


class ChatMessage(TypedDict):
    """A message displayed in the assistant interface."""

    role: Literal["user", "assistant"]
    content: str


def initialize_assistant_state() -> None:
    """Initialize the assistant session state."""
    if THREAD_ID_KEY not in st.session_state:
        st.session_state[THREAD_ID_KEY] = str(uuid4())

    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []

    if PENDING_PROMPT_KEY not in st.session_state:
        st.session_state[PENDING_PROMPT_KEY] = None

    if FAILED_PROMPT_KEY not in st.session_state:
        st.session_state[FAILED_PROMPT_KEY] = None


def start_new_conversation() -> None:
    """Clear the current conversation and create a new thread."""
    st.session_state[THREAD_ID_KEY] = str(uuid4())
    st.session_state[MESSAGES_KEY] = []
    st.session_state[PENDING_PROMPT_KEY] = None
    st.session_state[FAILED_PROMPT_KEY] = None


def queue_prompt(prompt: str) -> None:
    """Store a suggested prompt for submission.

    Args:
        prompt: The suggested prompt.
    """
    st.session_state[PENDING_PROMPT_KEY] = prompt


def render_hero() -> None:
    """Render the assistant-page title and description."""
    st.title("Research Assistant")

    st.markdown(
        """
        Ask scientific and technical questions, retrieve relevant research
        papers, and receive evidence-grounded answers with citations.
        """
    )


def render_conversation_controls() -> None:
    """Render controls for managing the current conversation."""
    _, button_column = st.columns(
        [0.8, 0.2],
        vertical_alignment="center",
    )

    with button_column:
        st.button(
            "New conversation",
            icon=":material/add_comment:",
            type="secondary",
            width="stretch",
            key="assistant-new-conversation-button",
            on_click=start_new_conversation,
        )


def render_message(message: ChatMessage) -> None:
    """Render a single chat message.

    Args:
        message: The chat message to render.
    """
    avatar = (
        ":material/person:"
        if message["role"] == "user"
        else ":material/smart_toy:"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar,
    ):
        st.markdown(message["content"])


def render_chat_history() -> None:
    """Render all messages in the current conversation."""
    messages: list[ChatMessage] = st.session_state[MESSAGES_KEY]

    for message in messages:
        render_message(message)


def render_empty_state() -> None:
    """Render the initial assistant state."""
    st.markdown(
        """
        <div class="assistant-empty-state">
            <h3>What would you like to investigate?</h3>
            <p>
                Ask a scientific question, request papers on a topic, or
                compare approaches from the research literature.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prompt_columns = st.columns(2)

    for index, (label, prompt) in enumerate(SUGGESTED_PROMPTS):
        with prompt_columns[index % 2]:
            st.button(
                label,
                icon=":material/arrow_forward:",
                help=prompt,
                width="stretch",
                key=f"assistant-suggested-prompt-{index}",
                on_click=queue_prompt,
                args=(prompt,),
            )


def process_prompt(user_prompt: str) -> None:
    """Send a prompt to the backend and store the response.

    Args:
        user_prompt: The prompt submitted by the user.
    """
    cleaned_prompt = user_prompt.strip()

    if not cleaned_prompt:
        return

    user_message: ChatMessage = {
        "role": "user",
        "content": cleaned_prompt,
    }

    st.session_state[MESSAGES_KEY].append(user_message)
    st.session_state[FAILED_PROMPT_KEY] = None

    render_message(user_message)

    try:
        with st.chat_message(
            "assistant",
            avatar=":material/smart_toy:",
        ):
            with st.spinner(
                "Searching the literature and preparing the answer...",
                show_time=True,
            ):
                assistant_response: AssistantResponse = chat(
                    user_query=cleaned_prompt,
                    thread_id=st.session_state[THREAD_ID_KEY],
                )

            st.markdown(assistant_response.response)

    except Exception:
        LOGGER.exception(
            "Assistant request failed for thread %s.",
            st.session_state[THREAD_ID_KEY],
        )

        # Remove the unanswered user message from the stored history.
        st.session_state[MESSAGES_KEY].pop()
        st.session_state[FAILED_PROMPT_KEY] = cleaned_prompt

        st.error(
            "The message request failed. Please try again.",
            icon=":material/error:",
        )

        return

    # The Pydantic UUID4 value is converted back to a string because chat()
    # currently accepts thread_id as str.
    st.session_state[THREAD_ID_KEY] = str(assistant_response.thread_id)

    assistant_message: ChatMessage = {
        "role": "assistant",
        "content": assistant_response.response,
    }

    st.session_state[MESSAGES_KEY].append(assistant_message)


def render_assistant_page() -> None:
    """Renders the assistant page."""
    initialize_assistant_state()

    render_hero()

    render_conversation_controls()

    submitted_prompt = st.chat_input(
        "Ask a scientific question or request relevant papers...",
        key="assistant-chat-input",
        max_chars=2_000,
        accept_audio=False,
        accept_file=False,
    )

    pending_prompt: str | None = st.session_state.pop(
        PENDING_PROMPT_KEY,
        None,
    )

    prompt_to_process = submitted_prompt or pending_prompt

    with st.container(
        height=400,
        border=True,
        key="assistant-chat-body",
    ):
        messages: list[ChatMessage] = st.session_state[MESSAGES_KEY]

        if messages:
            render_chat_history()
        elif prompt_to_process is None:
            render_empty_state()

        if isinstance(prompt_to_process, str):
            process_prompt(prompt_to_process)


render_assistant_page()
