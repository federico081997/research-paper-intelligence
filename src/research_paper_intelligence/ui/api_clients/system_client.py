"""Client for retrieving system information."""

import httpx2
import streamlit as st

from research_paper_intelligence.api.schemas.system import (
    SystemInfoResponse,
)
from research_paper_intelligence.config import get_settings


@st.cache_data(
    ttl=15,
    show_spinner=False,
)
def get_system_info() -> SystemInfoResponse:
    """Retrieves information about the loaded system resources."""
    settings = get_settings()

    with httpx2.Client(
        base_url="http://" + settings.api_host + ":" + str(settings.api_port),
        timeout=settings.api_timeout_seconds,
    ) as client:
        response = client.get("/api/v1/system/")
        response.raise_for_status()

    return SystemInfoResponse.model_validate(response.json())
