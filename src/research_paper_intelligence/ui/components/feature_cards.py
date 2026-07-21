"""Utilities for rendering feature cards."""

import streamlit as st


def render_feature_card(
    *,
    key: str,
    title: str,
    description: str,
    icon: str,
    status: str,
    page: str | None = None,
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
