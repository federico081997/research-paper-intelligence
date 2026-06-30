"""Generate and load vector embeddings for research-papers."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_model(model_name: str) -> SentenceTransformer:
    """Instantiate the SentenceTransformer model.

    Args:
        model_name: Name of the SentenceTransformer model.

    Returns:
        The initialized SentenceTransformer model.
    """
    model: SentenceTransformer = SentenceTransformer(model_name)

    return model


def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Generate embeddings for a collection of texts.

    Args:
        model: SentenceTransformer model.
        texts: Texts to encode.
        batch_size: Number of texts processed in each batch.
        device: Compute device used for encoding.

    Returns:
        Generated embedding matrix as a float32 NumPy array.

    Raises:
        ValueError: If no texts are provided.
    """
    if not texts:
        raise ValueError("At least one text is required.")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        device=str(device),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return np.asarray(embeddings, dtype=np.float32)
