"""Classificação semântica de perguntas do puzzle.

O classificador possui dois modos de decisão, descritos em ``src/config.py``:

* ``MODO_PADRAO`` (Abordagem A) reproduz o comportamento da Semana 1. A
  pergunta é comparada com todas as frases de referência do CSV, inclusive as
  rotuladas como ``TOKEN_NAO_RELACIONADO``, e vence a maior similaridade.
* ``MODO_REJEICAO`` (Abordagem B) compara a pergunta apenas com as referências
  das cinco intenções do puzzle. Se a maior similaridade não alcançar o limiar
  informado, a pergunta é rejeitada e recebe ``TOKEN_NAO_RELACIONADO``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    DATASET_PATH,
    MODEL_NAME,
    MODO_PADRAO,
    MODO_REJEICAO,
    MODOS_VALIDOS,
    SIMILARITY_THRESHOLD,
    TOKEN_NAO_RELACIONADO,
    TOP_K,
)
from .embeddings import EmbeddingService
from .similarity import matriz_similaridade, ordenar_indices


class PuzzleClassifier:
    """Classifica perguntas pela frase de referência semanticamente mais próxima.

    O modelo de embeddings é carregado uma única vez na construção da instância
    e os embeddings das frases de referência são calculados apenas nesse
    momento. Para cada nova pergunta, somente o embedding da pergunta é gerado.
    """

    def __init__(
        self,
        dataset_path: Path | str = DATASET_PATH,
        embedding_service: Optional[Any] = None,
        top_k: int = TOP_K,
        similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD,
    ) -> None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("TOP_K precisa ser um inteiro maior que zero.")
        _validar_threshold(similarity_threshold)

        self.dataset_path = Path(dataset_path)
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_service = embedding_service or EmbeddingService(MODEL_NAME)
        self.dataset = self._carregar_dataset()
        self.frases_referencia: List[str] = self.dataset["frase"].tolist()
        self.tokens_referencia: List[str] = self.dataset["token"].tolist()
        self.embeddings_referencia = np.asarray(
            self.embedding_service.gerar_embeddings(self.frases_referencia),
            dtype=float,
        )

        # Índices das referências que participam de cada modo de decisão.
        self.indices_padrao = np.arange(len(self.frases_referencia))
        self.indices_puzzle = np.asarray(
            [
                indice
                for indice, token in enumerate(self.tokens_referencia)
                if token != TOKEN_NAO_RELACIONADO
            ],
            dtype=int,
        )
        if self.indices_puzzle.size == 0:
            raise ValueError(
                "O dataset de referência não possui frases das intenções do puzzle."
            )

    # ------------------------------------------------------------------
    # Carregamento do dataset
    # ------------------------------------------------------------------
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
        except OSError as exc:
            raise ValueError(
                f"Não foi possível ler o dataset de referência: {exc}"
            ) from exc

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
        return dataset.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Similaridades
    # ------------------------------------------------------------------
    def matriz_similaridade(self, perguntas: Sequence[str]) -> np.ndarray:
        """Gera a matriz de similaridades entre perguntas e todas as referências.

        Os embeddings das perguntas são calculados em uma única chamada ao
        modelo, evitando recomputação quando a mesma lista precisa ser avaliada
        em vários modos ou com vários limiares.

        Returns:
            Matriz ``(n_perguntas, n_referencias)``.
        """
        perguntas_limpas = [_validar_pergunta(pergunta) for pergunta in perguntas]
        if not perguntas_limpas:
            raise ValueError("É necessário informar ao menos uma pergunta.")
        embeddings = self.embedding_service.gerar_embeddings(perguntas_limpas)
        return matriz_similaridade(embeddings, self.embeddings_referencia)

    # ------------------------------------------------------------------
    # Classificação
    # ------------------------------------------------------------------
    def classificar(
        self,
        pergunta: str,
        modo: str = MODO_PADRAO,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Classifica uma pergunta e retorna a melhor correspondência e o Top-K.

        Args:
            pergunta: texto em português digitado pelo jogador.
            modo: ``"padrao"`` ou ``"rejeicao"``.
            threshold: limiar aplicado nesta chamada. Quando ``None``, usa
                ``self.similarity_threshold``.

        Returns:
            Dicionário com ``pergunta``, ``token``, ``similaridade``,
            ``frase_referencia``, ``top_resultados``, ``token_intencao``,
            ``modo``, ``threshold`` e ``rejeitada``.
        """
        pergunta_limpa = _validar_pergunta(pergunta)
        embedding = self.embedding_service.gerar_embedding(pergunta_limpa)
        similaridades = matriz_similaridade(embedding, self.embeddings_referencia)[0]
        return self._decidir(pergunta_limpa, similaridades, modo, threshold)

    def classificar_em_lote(
        self,
        perguntas: Sequence[str],
        modo: str = MODO_PADRAO,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Classifica várias perguntas gerando os embeddings em um único lote."""
        perguntas_limpas = [_validar_pergunta(pergunta) for pergunta in perguntas]
        matriz = self.matriz_similaridade(perguntas_limpas)
        return self.classificar_de_matriz(perguntas_limpas, matriz, modo, threshold)

    def classificar_de_matriz(
        self,
        perguntas: Sequence[str],
        matriz: np.ndarray,
        modo: str = MODO_PADRAO,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Deriva as decisões a partir de uma matriz de similaridades já calculada.

        Permite reaproveitar os mesmos embeddings para avaliar vários modos e
        vários limiares sem chamar o modelo novamente.
        """
        matriz = np.asarray(matriz, dtype=float)
        if matriz.ndim != 2:
            raise ValueError("A matriz de similaridades precisa ser bidimensional.")
        if matriz.shape[0] != len(perguntas):
            raise ValueError(
                "A matriz de similaridades não corresponde à quantidade de perguntas."
            )
        if matriz.shape[1] != len(self.frases_referencia):
            raise ValueError(
                "A matriz de similaridades não corresponde às frases de referência."
            )
        return [
            self._decidir(str(pergunta), matriz[linha], modo, threshold)
            for linha, pergunta in enumerate(perguntas)
        ]

    def similaridade_maxima_puzzle(self, matriz: np.ndarray) -> np.ndarray:
        """Maior similaridade de cada pergunta frente às cinco intenções do puzzle.

        É o score utilizado na análise de thresholds e no gráfico de
        distribuição das similaridades.
        """
        matriz = np.asarray(matriz, dtype=float)
        if matriz.ndim != 2:
            raise ValueError("A matriz de similaridades precisa ser bidimensional.")
        return matriz[:, self.indices_puzzle].max(axis=1)

    # ------------------------------------------------------------------
    # Regra de decisão
    # ------------------------------------------------------------------
    def _decidir(
        self,
        pergunta: str,
        similaridades: np.ndarray,
        modo: str,
        threshold: Optional[float],
    ) -> Dict[str, Any]:
        """Aplica a regra de decisão do modo escolhido sobre um vetor de scores."""
        if modo not in MODOS_VALIDOS:
            raise ValueError(
                f"Modo inválido: {modo!r}. Use um de {list(MODOS_VALIDOS)}."
            )
        limiar = self.similarity_threshold if threshold is None else threshold
        _validar_threshold(limiar)
        if modo == MODO_REJEICAO and limiar is None:
            raise ValueError("O modo de rejeição exige um limiar de similaridade.")

        indices_validos = (
            self.indices_puzzle if modo == MODO_REJEICAO else self.indices_padrao
        )
        scores = np.asarray(similaridades, dtype=float)[indices_validos]
        ordem_local = ordenar_indices(scores)
        ordem_global = indices_validos[ordem_local]

        resultados = [
            {
                "frase": self.frases_referencia[int(indice)],
                "token": self.tokens_referencia[int(indice)],
                "similaridade": float(similaridades[int(indice)]),
            }
            for indice in ordem_global[: self.top_k]
        ]
        melhor = resultados[0]
        token_intencao = melhor["token"]
        token = token_intencao
        rejeitada = False

        # No modo padrão o limiar é opcional: com None, vence o maior valor,
        # exatamente como na Semana 1. No modo de rejeição o limiar é a própria
        # condição de aceitação da pergunta.
        if limiar is not None and melhor["similaridade"] < limiar:
            token = TOKEN_NAO_RELACIONADO
            rejeitada = True

        return {
            "pergunta": pergunta,
            "token": token,
            "similaridade": melhor["similaridade"],
            "frase_referencia": melhor["frase"],
            "top_resultados": resultados,
            "token_intencao": token_intencao,
            "modo": modo,
            "threshold": limiar,
            "rejeitada": rejeitada,
        }


def _validar_pergunta(pergunta: Any) -> str:
    """Garante que a pergunta é um texto não vazio e devolve a versão limpa."""
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise ValueError("A pergunta não pode ser vazia.")
    return pergunta.strip()


def _validar_threshold(valor: Optional[float]) -> None:
    """Valida o limiar de similaridade, que precisa ser None ou estar em [-1, 1]."""
    if valor is None:
        return
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError("O limiar de similaridade precisa ser numérico ou None.")
    if not -1.0 <= float(valor) <= 1.0:
        raise ValueError("O limiar de similaridade precisa estar entre -1 e 1.")
