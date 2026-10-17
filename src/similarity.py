"""Funções para comparação por similaridade de cosseno.

A similaridade de cosseno entre dois vetores A e B é definida por

    sim(A, B) = (A . B) / (||A|| * ||B||)

e varia entre -1 e 1. Como os embeddings de frases curtas em português
produzidos pelo modelo utilizado raramente assumem valores negativos, a faixa
observada na prática fica próxima do intervalo [0, 1].
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def matriz_similaridade(
    embeddings_perguntas: np.ndarray,
    embeddings_referencia: np.ndarray,
) -> np.ndarray:
    """Calcula a matriz de similaridades entre perguntas e referências.

    Args:
        embeddings_perguntas: matriz ``(n_perguntas, dimensao)`` ou um único
            vetor ``(dimensao,)``.
        embeddings_referencia: matriz ``(n_referencias, dimensao)``.

    Returns:
        Matriz ``(n_perguntas, n_referencias)`` com valores em ``[-1, 1]``.

    Raises:
        ValueError: se as referências estiverem vazias ou as dimensões não
            forem compatíveis.
    """
    perguntas = np.asarray(embeddings_perguntas, dtype=float)
    if perguntas.ndim == 1:
        perguntas = perguntas.reshape(1, -1)
    referencias = np.asarray(embeddings_referencia, dtype=float)

    if perguntas.ndim != 2 or perguntas.shape[0] == 0:
        raise ValueError("Os embeddings das perguntas não podem estar vazios.")
    if referencias.ndim != 2 or referencias.shape[0] == 0:
        raise ValueError("Os embeddings de referência não podem estar vazios.")
    if referencias.shape[1] != perguntas.shape[1]:
        raise ValueError("Os embeddings precisam ter a mesma dimensão.")

    valores = cosine_similarity(perguntas, referencias)
    return np.clip(valores, -1.0, 1.0)


def comparar_embeddings(
    embedding_pergunta: np.ndarray,
    embeddings_referencia: np.ndarray,
) -> List[Tuple[int, float]]:
    """Retorna índices e similaridades ordenados da maior para a menor."""
    valores = matriz_similaridade(embedding_pergunta, embeddings_referencia)[0]
    indices_ordenados = ordenar_indices(valores)
    return [(int(indice), float(valores[indice])) for indice in indices_ordenados]


def ordenar_indices(valores: np.ndarray) -> np.ndarray:
    """Ordena índices por similaridade decrescente de forma determinística.

    Empates são resolvidos pelo menor índice, o que mantém a ordem original do
    CSV de referência e torna o ranking reprodutível entre execuções.
    """
    valores = np.asarray(valores, dtype=float)
    return np.lexsort((np.arange(valores.size), -valores))
