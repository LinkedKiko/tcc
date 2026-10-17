"""Teste de integração com o modelo real de embeddings.

Este arquivo é o único que carrega o SentenceTransformer. Ele é ignorado
automaticamente quando a biblioteca não está instalada ou quando o modelo não
pode ser carregado (por exemplo, sem acesso à internet na primeira execução).

Para rodar somente este teste::

    python -m pytest tests/test_integracao_modelo.py -m integracao

Para rodar a suíte sem ele::

    python -m pytest -m "not integracao"

O teste verifica apenas propriedades estruturais (dimensão, intervalo das
similaridades, consistência entre lote e chamada individual). Ele não afirma
que determinada frase deve atingir um score específico, pois esses valores
dependem da versão do modelo.
"""

from __future__ import annotations

import pytest

from src.classifier import PuzzleClassifier
from src.config import DATASET_PATH, MODEL_NAME

pytest.importorskip("sentence_transformers", reason="sentence-transformers não instalado")

pytestmark = pytest.mark.integracao


@pytest.fixture(scope="module")
def classificador_real() -> PuzzleClassifier:
    """Carrega o modelo real uma única vez para todo o módulo."""
    try:
        return PuzzleClassifier(DATASET_PATH)
    except Exception as exc:  # pragma: no cover - depende do ambiente
        pytest.skip(f"Não foi possível carregar o modelo {MODEL_NAME}: {exc}")


def test_modelo_real_gera_embeddings_com_dimensao_consistente(
    classificador_real: PuzzleClassifier,
) -> None:
    assert classificador_real.embeddings_referencia.shape[0] == 90
    assert classificador_real.embeddings_referencia.shape[1] > 0


def test_modelo_real_classifica_e_devolve_similaridade_valida(
    classificador_real: PuzzleClassifier,
) -> None:
    resultado = classificador_real.classificar(
        "O que significam os símbolos ao lado da porta?"
    )

    assert resultado["token"] in set(classificador_real.tokens_referencia)
    assert -1.0 <= resultado["similaridade"] <= 1.0
    assert len(resultado["top_resultados"]) == classificador_real.top_k


def test_modelo_real_em_lote_concorda_com_chamadas_individuais(
    classificador_real: PuzzleClassifier,
) -> None:
    perguntas = [
        "Preciso contar os animais do mural?",
        "Como abro o cofre?",
    ]

    em_lote = classificador_real.classificar_em_lote(perguntas)
    individuais = [classificador_real.classificar(p) for p in perguntas]

    assert [r["token"] for r in em_lote] == [r["token"] for r in individuais]
    for esperado, obtido in zip(individuais, em_lote):
        assert obtido["similaridade"] == pytest.approx(esperado["similaridade"], abs=1e-5)
