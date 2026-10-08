"""Geração de embeddings com Sentence Transformers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class EmbeddingService:
    """Carrega um modelo de embeddings e o reutiliza durante a execução."""

    def __init__(self, model_name: str) -> None:
        # A importação tardia permite testar a lógica do classificador com um
        # serviço falso sem exigir o download do modelo nos testes unitários.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def gerar_embedding(self, frase: str) -> np.ndarray:
        """Gera o embedding de uma única frase."""
        if not isinstance(frase, str) or not frase.strip():
            raise ValueError("A frase precisa ser um texto não vazio.")

        embedding = self.model.encode(frase, convert_to_numpy=True)
        return np.asarray(embedding, dtype=float)

    def gerar_embeddings(self, frases: Sequence[str]) -> np.ndarray:
        """Gera os embeddings de várias frases em uma única chamada."""
        if not frases:
            raise ValueError("É necessário informar ao menos uma frase.")
        if any(not isinstance(frase, str) or not frase.strip() for frase in frases):
            raise ValueError("Todas as frases precisam ser textos não vazios.")

        embeddings = self.model.encode(list(frases), convert_to_numpy=True)
        return np.asarray(embeddings, dtype=float)
