"""Global styles for the Streamlit application."""

import streamlit as st


def apply_app_styles() -> None:
    """Apply application-wide custom CSS."""
    st.html(
        """
        <style>
        div[class*="st-key-feature-card-"] {
            min-height: 15rem;
            padding: 1.35rem;
            background:
                linear-gradient(
                    145deg,
                    #ffffff 0%,
                    #eff6ff 100%
                );
            border: 1px solid #bfdbfe;
            border-radius: 0.9rem;
            box-shadow:
                0 2px 5px rgba(15, 23, 42, 0.04),
                0 8px 20px rgba(37, 99, 235, 0.07);
            transition:
                transform 150ms ease,
                border-color 150ms ease,
                box-shadow 150ms ease;
        }

        div[class*="st-key-feature-card-"]:hover {
            transform: translateY(-3px);
            border-color: #60a5fa;
            box-shadow:
                0 4px 8px rgba(15, 23, 42, 0.06),
                0 12px 26px rgba(37, 99, 235, 0.12);
        }

        div[class*="st-key-feature-card-"] h3 {
            color: #1e3a8a;
            margin-bottom: 1rem;
        }

        div[class*="st-key-feature-card-"] p {
            color: #475569;
            line-height: 1.6;
        }
        </style>
        """
    )
