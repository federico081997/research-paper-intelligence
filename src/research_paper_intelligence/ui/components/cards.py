"""Utilities for rendering application cards."""

from collections.abc import Callable

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from research_paper_intelligence.api.schemas.search import (
    SearchResponse,
    SearchResultItem,
)


def render_feature_card(
    *,
    key: str,
    title: str,
    description: str,
    icon: str,
    status: str,
    page: StreamlitPage | None = None,
    button_label: str = "Open feature",
) -> None:
    """Render an application feature card.

    Args:
        key: The key of the feature card.
        title: The title of the feature.
        description: The description of the feature.
        icon: The icon to display.
        status: The status of the feature.
        page: The page to link to.
        button_label: The label of the link button.
    """
    with st.container(key=f"feature-card-{key}"):
        st.markdown(f"### {icon} {title}")
        st.write(description)
        st.caption(status)

        if page is not None:
            st.page_link(
                page,
                label=button_label,
                icon=":material/arrow_forward:",
                width="stretch",
            )
        else:
            st.button(
                "Coming soon",
                disabled=True,
                width="stretch",
            )


def render_result_card(
    result: SearchResultItem,
    selected_paper_id_key: str,
    on_click: Callable[[str], None],
) -> None:
    """Render one selectable search-result card.

    Args:
        result: Search result to display.
        selected_paper_id_key: The session state key of the selected paper.
        on_click: Callback function that will be called when clicked.
    """
    is_selected = result.paper_id == st.session_state[selected_paper_id_key]

    with st.container(
        key=f"result-card-{result.paper_id}",
        border=True,
    ):
        button_key = (
            f"selected-result-title-{result.paper_id}"
            if is_selected
            else f"result-title-{result.paper_id}"
        )
        st.button(
            label=f"**#{result.rank} - {result.title}**",
            key=button_key,
            type="tertiary",
            width="stretch",
            on_click=on_click,
            args=(result.paper_id,),
        )

        st.caption(
            f"{result.authors}  ·  "
            f"{result.category}  ·  "
            f"{result.published_date:%Y-%m-%d}"
        )

        st.caption(f"Hybrid match score: {result.hybrid_score:.3f}")


def render_result_list_card(
    response: SearchResponse,
    selected_paper_id_key: str,
    on_click: Callable[[str], None],
) -> None:
    """Render the scrollable container with the result list.

    Args:
        response: Search response to display.
        selected_paper_id_key: The session state key of the selected paper.
        on_click: Callback function that will be called when clicked.
    """
    st.subheader("Search results")

    st.caption(
        f"{response.total} papers returned in "
        f"{response.time_elapsed:.3f} seconds"
    )

    with st.container(
        height=760,
        key="result-list-body",
        border=False,
    ):
        for result in response.results:
            render_result_card(result, selected_paper_id_key, on_click)
