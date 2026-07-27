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
        
        .st-key-assistant-chat-body {
        padding: 1rem 1.1rem;
        background:
            linear-gradient(
                145deg,
                #ffffff 0%,
                #f8fbff 100%
            );
        border-color: #bfdbfe !important;
        border-radius: 0.9rem;
        box-shadow:
            0 2px 5px rgba(15, 23, 42, 0.04),
            0 8px 20px rgba(37, 99, 235, 0.07);
    }
    
    /* Individual chat messages */
    .st-key-assistant-chat-body div[data-testid="stChatMessage"] {
        margin-bottom: 0.85rem;
        padding: 1rem 1.1rem;
        border: 1px solid #dbeafe;
        border-radius: 0.85rem;
        background: rgba(255, 255, 255, 0.86);
    }
    
    /* Markdown spacing inside assistant messages */
    .st-key-assistant-chat-body div[data-testid="stChatMessageContent"] p {
        line-height: 1.65;
    }
    
    .st-key-assistant-chat-body div[data-testid="stChatMessageContent"]
    :is(h1, h2, h3, h4) {
        color: #1e3a8a;
        margin-top: 1.1rem;
    }
    
    /* Initial empty state */
    .assistant-empty-state {
        padding: 2.3rem 1rem 1.5rem;
        text-align: center;
    }
    
    .assistant-empty-state h3 {
        margin-bottom: 0.5rem;
        color: #1e3a8a;
    }
    
    .assistant-empty-state p {
        max-width: 650px;
        margin: 0 auto 1.3rem;
        color: #475569;
        line-height: 1.6;
    }
    
    /* Chat input interaction */
    div[class*="st-key-assistant-chat-input"] button {
        transition:
            transform 100ms ease,
            filter 120ms ease;
    }
    
    div[class*="st-key-assistant-chat-input"] button:active {
        transform: scale(0.94);
        filter: brightness(0.95);
    }
    
    /* New conversation button */
    div[class*="st-key-assistant-new-conversation-button"] button {
        width: 100%;
        padding: 0.7rem 1rem;
    
        color: #1e3a8a;
        font-weight: 600;
    
        background:
            linear-gradient(
                145deg,
                #ffffff 0%,
                #eff6ff 100%
            );
    
        border: 1px solid #bfdbfe;
        border-radius: 0.75rem;
    
        box-shadow:
            0 2px 5px rgba(15, 23, 42, 0.04),
            0 6px 16px rgba(37, 99, 235, 0.07);
    
        transition:
            transform 150ms ease,
            border-color 150ms ease,
            box-shadow 150ms ease;
    }
    
    div[class*="st-key-assistant-new-conversation-button"] button:hover {
        color: #2563eb;
        border-color: #60a5fa;
    
        transform: translateY(-2px);
    
        box-shadow:
            0 4px 8px rgba(15, 23, 42, 0.06),
            0 10px 22px rgba(37, 99, 235, 0.12);
    }
    
    div[class*="st-key-assistant-new-conversation-button"] button:active {
        transform: translateY(0) scale(0.98);
    
        box-shadow:
            0 1px 3px rgba(15, 23, 42, 0.08),
            0 3px 8px rgba(37, 99, 235, 0.08);
    }
    </style>
        """
    )
