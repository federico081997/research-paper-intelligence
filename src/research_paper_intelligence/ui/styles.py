"""Global styles for the Streamlit application."""

import streamlit as st


def apply_app_styles() -> None:
    """Apply application-wide custom CSS."""
    st.html(
        """
        <style>
        div[class*="st-key-feature-card-"],
        div[class*="st-key-result-details-card-"],
        div[class*="st-key-result-card-"] {
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

        div[class*="st-key-feature-card-"]:hover,
        div[class*="st-key-result-details-card-"]:hover,
        div[class*="st-key-result-card-"]:hover {
            transform: translateY(-3px);
            border-color: #60a5fa;
            box-shadow:
                0 4px 8px rgba(15, 23, 42, 0.06),
                0 12px 26px rgba(37, 99, 235, 0.12);
        }

        div[class*="st-key-feature-card-"],
        div[class*="st-key-result-details-card-"] h3 {
            color: #1e3a8a;
            margin-bottom: 1rem;
        }

        div[class*="st-key-feature-card-"],
        div[class*="st-key-result-details-card-"] p {
            color: #475569;
            line-height: 1.6;
        }

        div[class*="st-key-search-submit-button"] button {
            transition:
                transform 100ms ease,
                box-shadow 120ms ease,
                filter 120ms ease;
        }

        div[class*="st-key-search-submit-button"] button:active {
            transform: translateY(1px) scale(0.97);
            box-shadow:
                0 1px 2px rgba(15, 23, 42, 0.12),
                0 2px 6px rgba(37, 99, 235, 0.12);
            filter: brightness(0.96);
        }

        @media (prefers-reduced-motion: reduce) {
            div[class*="st-key-search-submit-button"] button {
                transition: none;
            }
        }
        
        .st-key-result-list-body {
            padding: 0.75rem 0.85rem 1rem 0.85rem;
            box-sizing: border-box;
        }

        .st-key-result-list-body div[class*="st-key-result-card-"] {
            margin-bottom: 0.8rem;
        }
        
        /* All clickable result titles */
        div[class*="st-key-result-title-"] button,
        div[class*="st-key-selected-result-title-"] button {
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            padding: 0.35rem 0;
            border: none;
            background: transparent;
            box-shadow: none;
        }
        
        /* Allow long paper titles to wrap */
        div[class*="st-key-result-title-"] button p,
        div[class*="st-key-selected-result-title-"] button p {
            width: 100%;
            text-align: center;
            white-space: normal;
            line-height: 1.4;
        }
        
        /* Hover effect */
        div[class*="st-key-result-title-"] button:hover,
        div[class*="st-key-selected-result-title-"] button:hover {
            border: none;
            background: transparent;
            box-shadow: none;
        }
        
        div[class*="st-key-result-title-"] button:hover p {
            color: #2563eb;
        }
        
        /* Selected title only */
        div[class*="st-key-selected-result-title-"] button {
            padding-left: 0.75rem;
            border-left: 4px solid #2563eb;
            border-radius: 0;
        }
        
        div[class*="st-key-selected-result-title-"] button p {
            color: #2563eb;
            font-weight: 700;
        }
        </style>
        """
    )
