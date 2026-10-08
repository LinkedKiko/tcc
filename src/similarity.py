"""Funções para comparação por similaridade de cosseno."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def comparar_embeddings(
    embedding_pergunta: np.ndarray,
    embeddings_referencia: np.ndarray,
) -> List[Tuple[int, float]]:
    """Retorna índices e similaridades ordenados da maior para a menor."""
    pergunta = np.asarray(embedding_pergunta, dtype=float).reshape(1, -1)
    referencias = np.asarray(embeddings_referencia, dtype=float)

    if referencias.ndim != 2 or referencias.shape[0] == 0:
        raise ValueError("Os embeddings de referência não podem estar vazios.")
    if referencias.shape[1] != pergunta.shape[1]:
        raise ValueError("Os embeddings precisam ter a mesma dimensão.")

    valores = cosine_similarity(pergunta, referencias)[0]
    valores = np.clip(valores, -1.0, 1.0)
    indices_ordenados = np.argsort(-valores)
    return [(int(indice), float(valores[indice])) for indice in indices_ordenados]
