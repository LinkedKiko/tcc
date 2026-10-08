"""Classificação semântica de perguntas do puzzle."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import DATASET_PATH, MODEL_NAME, SIMILARITY_THRESHOLD, TOP_K
from .embeddings import EmbeddingService
from .similarity import comparar_embeddings


class PuzzleClassifier:
    """Classifica perguntas pela frase de referência semanticamente mais próxima."""

    def __init__(
        self,
        dataset_path: Path | str = DATASET_PATH,
        embedding_service: Optional[Any] = None,
        top_k: int = TOP_K,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> None:
        if top_k <= 0:
            raise ValueError("TOP_K precisa ser maior que zero.")

        self.dataset_path = Path(dataset_path)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_service = embedding_service or EmbeddingService(MODEL_NAME)
        self.dataset = self._carregar_dataset()
        self.frases_referencia = self.dataset["frase"].tolist()
        self.tokens_referencia = self.dataset["token"].tolist()
        self.embeddings_referencia = self.embedding_service.gerar_embeddings(
            self.frases_referencia
        )

    def _carregar_dataset(self) -> pd.DataFrame:
        """Carrega e valida as colunas e os registros do CSV de referência."""
        try:
            dataset = pd.read_csv(self.dataset_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Dataset de referência não encontrado: {self.dataset_path}"
            ) from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError("O dataset de referência está vazio.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"Não foi possível interpretar o CSV: {exc}") from exc

        colunas_obrigatorias = {"frase", "token"}
        if not colunas_obrigatorias.issubset(dataset.columns):
            faltantes = sorted(colunas_obrigatorias - set(dataset.columns))
            raise ValueError(f"Colunas obrigatórias ausentes no CSV: {faltantes}")
        if dataset.empty:
            raise ValueError("O dataset de referência não possui registros.")
        if dataset[["frase", "token"]].isnull().any().any():
            raise ValueError("As colunas frase e token não podem possuir valores vazios.")

        dataset = dataset[["frase", "token"]].copy()
        dataset["frase"] = dataset["frase"].astype(str).str.strip()
        dataset["token"] = dataset["token"].astype(str).str.strip()
        if (dataset["frase"] == "").any() or (dataset["token"] == "").any():
            raise ValueError("As colunas frase e token não podem possuir textos vazios.")
        return dataset

    def classificar(self, pergunta: str) -> Dict[str, Any]:
        """Classifica uma pergunta e retorna a melhor correspondência e o Top-K."""
        if not isinstance(pergunta, str) or not pergunta.strip():
            raise ValueError("A pergunta não pode ser vazia.")

        pergunta = pergunta.strip()
        embedding_pergunta = self.embedding_service.gerar_embedding(pergunta)
        ranking = comparar_embeddings(embedding_pergunta, self.embeddings_referencia)
        resultados = [
            {
                "frase": self.frases_referencia[indice],
                "token": self.tokens_referencia[indice],
                "similaridade": similaridade,
            }
            for indice, similaridade in ranking[: self.top_k]
        ]
        melhor = resultados[0]
        token = melhor["token"]

        # O limiar permanece opcional para futuros experimentos. Não há um
        # valor arbitrário nesta primeira semana; com None, vence o maior valor.
        if (
            self.similarity_threshold is not None
            and melhor["similaridade"] < self.similarity_threshold
        ):
            token = "TOKEN_NAO_RELACIONADO"

        return {
            "pergunta": pergunta,
            "token": token,
            "similaridade": melhor["similaridade"],
            "frase_referencia": melhor["frase"],
            "top_resultados": resultados,
        }
