from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    """
    Carrega o Sentence Transformer uma única vez (cache em memória).
    """
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    """
    Gera embeddings normalizados para uma lista de sentenças.
    """
    encoder = get_encoder()
    return encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
