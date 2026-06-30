"""Coordinate research-paper embedding generation."""

import logging
from pathlib import Path

import pandas as pd

from research_paper_intelligence.config import Settings
from research_paper_intelligence.data.data_loader import load_data
from research_paper_intelligence.device_config import get_device
from research_paper_intelligence.embeddings.encoder import (
    generate_embeddings,
    get_model,
)
from research_paper_intelligence.storage.embedding_io import (
    save_embeddings,
)

logger = logging.getLogger(__name__)


def extract_embedding_texts(dataframe: pd.DataFrame) -> list[str]:
    """Extract valid texts which include the title and summary combined.

    Args:
        dataframe: Processed research-paper DataFrame.

    Returns:
        Texts used to generate embeddings.

    Raises:
        KeyError: If the "combined_text" column is missing.
        ValueError: If the column contains missing or empty values.
    """
    if not {"title", "summary"}.issubset(dataframe.columns):
        raise KeyError("Required columns are missing: title, summary.")

    combined_text = (
        dataframe["title"].fillna("").astype(str).str.strip()
        + " "
        + dataframe["summary"].fillna("").astype(str).str.strip()
    ).str.strip()

    return combined_text.tolist()


def run_embedding_pipeline(settings: Settings) -> Path:
    """Load paper text, generate embeddings, and save them.

    Args:
        settings: Validated application configuration.

    Returns:
        Path containing the saved embedding matrix.
    """
    # Load the processed data.
    logger.info(
        "Loading processed papers from %s",
        settings.processed_papers_path,
    )
    papers = load_data(settings.processed_papers_path)

    # Extract the text to be embedded from the 'combined_text' column.
    logger.info(
        "Extracting text from %d abstracts",
        len(papers),
    )
    texts = extract_embedding_texts(papers)

    # Set up the current compute device.
    device = get_device(settings.device)

    # Initialize the embedding model
    logger.info(
        "Loading embedding model: %s",
        settings.embedding_model_name,
    )
    model = get_model(settings.embedding_model_name)

    # Generate the embeddings.
    logger.info(
        "Generating embeddings for %d abstracts",
        len(texts),
    )
    embeddings = generate_embeddings(
        model=model,
        texts=texts,
        batch_size=settings.embedding_batch_size,
        device=device,
    )

    # The number of embeddings should match the number of abstracts.
    if embeddings.shape[0] != len(papers):
        raise RuntimeError(
            "The number of generated embeddings does not match "
            "the number of papers."
        )

    # Save the embeddings to disk using a .npy extension
    logger.info(
        "Saving embedding matrix with shape %s to %s",
        embeddings.shape,
        settings.paper_embeddings_path,
    )
    save_embeddings(
        embeddings=embeddings,
        path=settings.paper_embeddings_path,
    )

    return settings.paper_embeddings_path
