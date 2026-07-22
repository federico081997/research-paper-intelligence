"""Home page for the Research Paper Intelligence application."""

import httpx2
import streamlit as st
from pydantic import ValidationError

from research_paper_intelligence.api.schemas.system import SystemInfoResponse
from research_paper_intelligence.ui.api_clients.system_client import (
    get_system_info,
)
from research_paper_intelligence.ui.components.cards import (
    render_feature_card,
)
from research_paper_intelligence.ui.navigation import (
    ASSISTANT_PAGE,
    SEARCH_PAGE,
)


def render_hero() -> None:
    """Render the hero section of the home page."""
    st.title("Research Paper Intelligence")

    st.markdown(
        """
        Discover and rank research papers using hybrid retrieval,
        then analyse them with evidence-grounded AI workflows.
        """
    )


def render_system_metrics(system_info: SystemInfoResponse) -> None:
    """Render the main system summary values.

    Args:
        system_info: System information.
    """
    paper_column, retrieval_column, index_column, version_column = st.columns(
        [1, 1, 1, 1]
    )

    with paper_column:
        st.metric(
            label="Available papers",
            value=f"{system_info.paper_count:,.0f}",
        )

    with retrieval_column:
        st.metric(
            label="Retrieval method", value=system_info.retrieval_strategy
        )

    with index_column:
        st.metric(
            label="Indexed papers",
            value=f"{system_info.faiss_index_size:,}",
        )

    with version_column:
        st.metric(label="API version", value=system_info.api_version)


def render_retrieval_details(system_info: SystemInfoResponse) -> None:
    """Render detailed information about the retrieval system.

    Args:
        system_info: System information.
    """
    with st.expander("**Retrieval configuration**"):
        components_column, resources_column = st.columns(2)

        with components_column:
            st.markdown("#### Hybrid ranking components")
            for component in system_info.ranking_components:
                st.markdown(f"- {component}")

        with resources_column:
            st.markdown("#### Loaded resources")

            st.markdown(
                f"""
                **Embedding model**

                `{system_info.embedding_model}`

                **FAISS index type**

                `{system_info.faiss_index_type}`

                **TF-IDF documents**

                `{system_info.tfidf_document_count:,}`

                **TF-IDF vocabulary**

                `{system_info.tfidf_vocabulary_size:,}`
                """
            )


def render_system_status() -> None:
    """Load and display the current backend status."""
    st.subheader("System status")

    try:
        system_info = get_system_info()

    except httpx2.ConnectError:
        st.error(
            "The FastAPI backend is unavailable. "
            "Start the API server and refresh this page."
        )
        return

    except httpx2.TimeoutException:
        st.warning(
            "The backend did not respond in time. "
            "It may still be loading the search resources."
        )
        return

    except httpx2.HTTPStatusError as error:
        st.error(
            "The backend returned an error with status "
            f"{error.response.status_code}."
        )
        return

    except ValidationError:
        st.error(
            "The backend returned system information in an unexpected format."
        )
        return

    if system_info.status == "ready":
        st.success(
            "The paper data, embedding model, FAISS index, "
            "and TF-IDF resources are ready."
        )
    else:
        st.warning(
            "The backend is running, but one or more search "
            "resources are not ready."
        )

    render_system_metrics(system_info)
    render_retrieval_details(system_info)


def render_capabilities() -> None:
    """Render the main application capabilities."""
    st.subheader("What you can do")

    search_column, assistant_column = st.columns(2)

    with search_column:
        render_feature_card(
            key="hybrid-search",
            title="Paper search",
            icon=":material/search:",
            description=(
                "Retrieve and rank papers using semantic similarity, "
                "TF-IDF, keyword overlap, and publication recency."
            ),
            status="Available",
            page=SEARCH_PAGE,
            button_label="Search papers",
        )

    with assistant_column:
        render_feature_card(
            key="research-assistant",
            title="Research assistant",
            icon=":material/smart_toy:",
            description=(
                "Ask research questions and receive answers grounded "
                "in retrieved papers with traceable citations."
            ),
            status="Planned",
            page=ASSISTANT_PAGE,
            button_label="Ask questions",
        )


def render_scope_and_limitations() -> None:
    """Render the application's current limitations."""
    with st.expander("**Scope and limitations**"):
        st.markdown(
            """
            - Search is limited to papers contained in the indexed
              dataset.
            - A high hybrid score indicates estimated relevance,
              not scientific correctness.
            - Ranking quality depends on the search query and the
              available paper metadata.
            - Future AI-generated answers should be verified against
              their cited source papers.
            """
        )


def render_home_page() -> None:
    """Render the complete Home page."""
    render_hero()

    st.divider()

    render_system_status()

    st.divider()

    render_capabilities()

    st.divider()

    render_scope_and_limitations()


render_home_page()
