"""Fixtures compartilhadas pelos testes.

Os testes unitários não baixam nem carregam o modelo SentenceTransformer: o
serviço de embeddings é substituído por um serviço determinístico de seis
dimensões, cada uma associada a um grupo de palavras do puzzle. Isso permite
testar as regras de negócio (ordenação, limiar, rejeição, métricas) sem
depender de scores semânticos reais.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "frases_referencia.csv"
AVALIACAO_PATH = PROJECT_ROOT / "data" / "teste_manual.csv"


class FakeEmbeddingService:
    """Serviço determinístico para testar a lógica sem carregar o modelo real.

    Cada dimensão conta quantos marcadores de um grupo aparecem no texto. A
    última dimensão representa assuntos externos ao puzzle. Textos sem nenhum
    marcador recebem um vetor constante, de modo que toda frase possui norma
    diferente de zero e a similaridade de cosseno permanece definida.
    """

    GRUPOS = {
        0: ("símbol", "simbol", "parede", "porta", "desenho", "painel", "escrit"),
        1: ("livro", "enciclop", "página", "pagina", "dicionár", "dicionar"),
        2: ("mural", "contar", "quantidade", "figura", "animal", "criatura"),
        3: ("cofre", "caixa", "cadeado", "compartimento", "fotografia"),
        4: ("relógio", "relogio", "horário", "horario", "hora", "ponteiro", "minuto"),
        5: ("capital", "bolo", "filme", "planeta", "receita", "futebol", "música"),
    }

    def _embedding(self, texto: str) -> np.ndarray:
        texto = texto.casefold()
        vetor = np.zeros(len(self.GRUPOS), dtype=float)
        for indice, marcadores in self.GRUPOS.items():
            vetor[indice] = sum(marcador in texto for marcador in marcadores)
        if not vetor.any():
            vetor[:] = 1.0
        return vetor

    def gerar_embedding(self, frase: str) -> np.ndarray:
        if not isinstance(frase, str) or not frase.strip():
            raise ValueError("A frase precisa ser um texto não vazio.")
        return self._embedding(frase)

    def gerar_embeddings(self, frases: Sequence[str]) -> np.ndarray:
        if not list(frases):
            raise ValueError("É necessário informar ao menos uma frase.")
        return np.vstack([self.gerar_embedding(frase) for frase in frases])


@pytest.fixture
def fake_embeddings() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def caminho_referencia() -> Path:
    return DATASET_PATH


@pytest.fixture
def caminho_avaliacao() -> Path:
    return AVALIACAO_PATH


@pytest.fixture
def classificador(fake_embeddings: FakeEmbeddingService):
    """Classificador real sobre o CSV real, com embeddings determinísticos."""
    from src.classifier import PuzzleClassifier

    return PuzzleClassifier(
        dataset_path=DATASET_PATH,
        embedding_service=fake_embeddings,
    )


@pytest.fixture
def csv_referencia_minimo(tmp_path: Path) -> Path:
    """CSV de referência reduzido, com duas frases por token do puzzle."""
    linhas = []
    marcadores = {
        "TOKEN_SIMBOLOS": "simbolo parede",
        "TOKEN_LIVRO": "livro enciclopedia",
        "TOKEN_MURAL": "mural contar",
        "TOKEN_COFRE": "cofre caixa",
        "TOKEN_RELOGIO": "relogio hora",
        "TOKEN_NAO_RELACIONADO": "capital planeta",
    }
    for token, texto in marcadores.items():
        for repeticao in range(2):
            linhas.append({"frase": f"{texto} pergunta {repeticao}", "token": token})
    caminho = tmp_path / "referencia.csv"
    pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")
    return caminho


@pytest.fixture
def dataset_avaliacao_minimo() -> pd.DataFrame:
    """Conjunto rotulado pequeno, usado nos testes do módulo de avaliação."""
    return pd.DataFrame(
        [
            {
                "frase": "O que significam os simbolos da parede?",
                "token_esperado": "TOKEN_SIMBOLOS",
                "dificuldade": "facil",
            },
            {
                "frase": "O livro ajuda a traduzir?",
                "token_esperado": "TOKEN_LIVRO",
                "dificuldade": "facil",
            },
            {
                "frase": "Preciso contar os bichos do mural?",
                "token_esperado": "TOKEN_MURAL",
                "dificuldade": "media",
            },
            {
                "frase": "Como abro o cofre?",
                "token_esperado": "TOKEN_COFRE",
                "dificuldade": "facil",
            },
            {
                "frase": "O relogio mostra uma hora estranha?",
                "token_esperado": "TOKEN_RELOGIO",
                "dificuldade": "facil",
            },
            {
                "frase": "Qual e a capital do Japao?",
                "token_esperado": "TOKEN_NAO_RELACIONADO",
                "dificuldade": "facil",
            },
        ]
    )
