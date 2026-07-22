"""Search page for the Research Paper Intelligence application."""

import logging

import httpx2
import streamlit as st
from pydantic import ValidationError

from research_paper_intelligence.api.schemas.search import (
    SearchResponse,
    SearchResultItem,
)
from research_paper_intelligence.ui.api_clients.search_client import (
    search_papers,
)
from research_paper_intelligence.ui.components.cards import (
    render_result_list_card,
)

SEARCH_RESPONSE_KEY = "paper_search_response"
SELECTED_PAPER_ID_KEY = "selected_paper_id"
LAST_SEARCH_QUERY_KEY = "last_search_query"

LOGGER = logging.getLogger(__name__)


def initialize_search_state() -> None:
    """Initialize values used by the search page."""
    if SEARCH_RESPONSE_KEY not in st.session_state:
        st.session_state[SEARCH_RESPONSE_KEY] = None

    if SELECTED_PAPER_ID_KEY not in st.session_state:
        st.session_state[SELECTED_PAPER_ID_KEY] = None

    if LAST_SEARCH_QUERY_KEY not in st.session_state:
        st.session_state[LAST_SEARCH_QUERY_KEY] = ""


def render_hero() -> None:
    """Render the search-page title and description."""
    st.title("Research Paper Search")

    st.markdown(
        """
        Retrieve and rank papers using semantic similarity, TF-IDF, 
        keyword overlap, and publication recency.
        """
    )


def render_search_form() -> tuple[bool, str, int]:
    """Render the paper-search form.

    Returns:
        A tuple containing whether the form was submitted, the entered
        search query and the requested number of results.
    """
    with st.container(
        key="search-form-card",
        border=True,
    ):
        st.markdown("### :material/search: Find research papers")

        st.caption(
            "Describe a topic, method, or research problem to retrieve "
            "and rank relevant arXiv papers."
        )

        with st.form(
            key="search-form",
            clear_on_submit=False,
            border=False,
        ):
            query_column, results_k_column, search_button_column = st.columns(
                [3, 1, 1],
                vertical_alignment="bottom",
            )

            with query_column:
                query = str(
                    st.text_input(
                        label="Search query",
                        placeholder="e.g. machine learning",
                        value="",
                        max_chars=500,
                    )
                )

            with results_k_column:
                result_k = st.number_input(
                    label="Number of results",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=5,
                )

                assert result_k is not None

            with search_button_column:
                submitted = st.form_submit_button(
                    label="Search papers",
                    type="primary",
                    key="search-submit-button",
                    icon=":material/search:",
                    width="stretch",
                )

    return submitted, query, int(result_k)


def perform_search(query: str, result_k: int) -> None:
    """Execute a search and save the results in session state.

    Args:
        query: Search query supplied by the user.
        result_k: Number of results to return.
    """
    try:
        with st.spinner("Searching and ranking papers..."):
            response = search_papers(query, result_k)

            st.session_state[SEARCH_RESPONSE_KEY] = response
            st.session_state[LAST_SEARCH_QUERY_KEY] = query.strip()

            # Automatically select the first paper by default
            st.session_state[SELECTED_PAPER_ID_KEY] = (
                response.results[0].paper_id if response.results else None
            )

    except httpx2.HTTPStatusError as exc:
        LOGGER.exception("Search API returned an HTTP error.")

        st.error(
            "The search service returned an error "
            f"({exc.response.status_code})"
        )

    except httpx2.RequestError:
        LOGGER.exception("Could not connect to the search API.")

        st.error(
            "Could not connect to the FastAPI backend. "
            "Make sure the API is running and try again."
        )

    except ValidationError:
        LOGGER.exception("The search API returned invalid data.")

        st.error("The backend returned data in an unexpected format.")


def build_arxiv_url(paper_id: str) -> str:
    """Build an arXiv abstract-page URL.

    Args:
        paper_id: A normalized arXiv identifier, such as
            ``cs/9308101v1`` or ``0901.4761v1``.

    Returns:
        The complete arXiv abstract-page URL.
    """
    return f"https://arxiv.org/abs/{paper_id}"


def select_paper(paper_id: str) -> None:
    """Update the session state with the selected paper id."""
    st.session_state[SELECTED_PAPER_ID_KEY] = paper_id


def get_selected_paper(
    response: SearchResponse,
) -> SearchResultItem | None:
    """Return the currently selected paper.

    Args:
        response: Search response containing the available papers.

    Returns:
        The selected paper, or None when no papers are available.
    """
    if not response.results:
        return None

    selected_paper_id = st.session_state[SELECTED_PAPER_ID_KEY]

    return next(
        (
            result
            for result in response.results
            if result.paper_id == selected_paper_id
        ),
        None,
    )


def render_score_summary(result: SearchResultItem) -> None:
    """Render the retrieval scores for a paper.

    Args:
        result: Selected search result.
    """
    first_row = st.columns(3)

    first_row[0].metric(
        label="Hybrid score",
        value=f"{result.hybrid_score:.3f}",
    )
    first_row[1].metric(
        label="Semantic",
        value=f"{result.semantic_score:.3f}",
    )
    first_row[2].metric(
        label="TF-IDF",
        value=f"{result.tfidf_score:.3f}",
    )

    second_row = st.columns(2)

    second_row[0].metric(
        label="Keyword overlap",
        value=f"{result.keyword_overlap_score:.3f}",
    )
    second_row[1].metric(
        label="Recency",
        value=f"{result.recency_score:.3f}",
    )


def render_paper_details(
    result: SearchResultItem | None,
) -> None:
    """Render the details of the selected paper.

    Args:
        result: Currently selected paper.
    """
    st.subheader("Paper details")

    if result is None:
        st.info("Select a paper to view its details.")
        return

    with st.container(
        key=f"result-details-card-{result.paper_id}",
        height=760,
    ):
        st.markdown(f"### {result.title}")

        st.caption(
            f"Rank {result.rank}  ·  "
            f"{result.category}  ·  "
            f"Published {result.published_date:%d %B %Y}"
        )

        st.markdown("#### Authors")
        st.write(result.authors)

        st.link_button(
            label="Open on arXiv",
            url=build_arxiv_url(result.paper_id),
            width="stretch",
        )

        st.divider()

        st.markdown("#### Abstract")
        st.write(result.abstract)

        st.divider()

        st.markdown("#### Why this paper matched")
        st.write(result.explanation)

        st.markdown("#### Ranking scores")
        render_score_summary(result)


def render_search_workspace(response: SearchResponse) -> None:
    """Render the result list and selected-paper details.

    Args:
        response: Search response to display.
    """
    if not response.results:
        st.info(
            "No papers matched this query. Try broader terminology "
            "or fewer constraints."
        )
        return

    results_column, details_column = st.columns(
        [1, 1],
        gap="small",
    )

    with results_column:
        render_result_list_card(
            response,
            selected_paper_id_key=SELECTED_PAPER_ID_KEY,
            on_click=select_paper,
        )

    with details_column:
        selected_paper = get_selected_paper(response)
        render_paper_details(selected_paper)


def render_search_page() -> None:
    """Render the complete search page."""
    initialize_search_state()
    render_hero()

    submitted, query, result_k = render_search_form()

    if submitted:
        if not query:
            return
        else:
            perform_search(
                query=query,
                result_k=result_k,
            )

    response = st.session_state[SEARCH_RESPONSE_KEY]

    if response is None:
        return

    st.caption(
        f'Showing results for: "{st.session_state[LAST_SEARCH_QUERY_KEY]}"'
    )

    render_search_workspace(response)


render_search_page()
