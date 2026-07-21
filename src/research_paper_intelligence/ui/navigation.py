"""Define Streamlit navigation pages."""

import streamlit as st

HOME_PAGE = st.Page(
    "pages/home.py",
    title="Home",
    icon=":material/home:",
    url_path="home",
    default=True,
)

SEARCH_PAGE = st.Page(
    "pages/search.py",
    title="Paper Search",
    icon=":material/search:",
    url_path="search",
)

ASSISTANT_PAGE = st.Page(
    "pages/assistant.py",
    title="Research Assistant",
    icon=":material/smart_toy:",
    url_path="assistant",
)

COMPARISON_PAGE = st.Page(
    "pages/comparison.py",
    title="Paper Comparison",
    icon=":material/compare_arrows:",
    url_path="comparison",
)

PAGES = {
    "Discover": [
        HOME_PAGE,
        SEARCH_PAGE,
        ASSISTANT_PAGE,
        COMPARISON_PAGE
    ]
}
