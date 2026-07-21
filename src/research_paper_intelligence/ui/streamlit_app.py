"""Main entry point for the Streamlit app."""

import streamlit as st

from research_paper_intelligence.ui.navigation import PAGES
from research_paper_intelligence.ui.styles import apply_app_styles

st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styles()

selected_page = st.navigation(
    PAGES,
    position="sidebar",
    expanded=True,
)

selected_page.run()
