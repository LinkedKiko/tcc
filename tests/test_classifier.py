"""Testes básicos do classificador sem depender do modelo remoto."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classifier import PuzzleClassifier
from src.responses import obter_dica


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "frases_referencia.csv"


class FakeEmbeddingService:
    """Serviço determinístico para testar a lógica sem carregar o modelo real."""

    def _embedding(self, texto: str) -> np.ndarray:
        texto = texto.casefold()
        palavras_por_indice = {
            0: ("símbol", "parede", "porta", "desenho"),
            1: ("livro", "enciclop", "página", "animal"),
            2: ("mural", "contar", "quantidade", "figura"),
            3: ("cofre", "senha", "combinação", "fotografia"),
            4: ("relógio", "horário", "hora", "ponteiro"),
            5: ("capital", "bolo", "filme", "planeta", "receita"),
        }
        vetor = np.zeros(6, dtype=float)
        for indice, palavras in palavras_por_indice.items():
            vetor[indice] = sum(palavra in texto for palavra in palavras)
        if not vetor.any():
            vetor[:] = 1
        return vetor

    def gerar_embedding(self, frase: str) -> np.ndarray:
        return self._embedding(frase)

    def gerar_embeddings(self, frases: list[str]) -> np.ndarray:
        return np.vstack([self._embedding(frase) for frase in frases])


def criar_classificador() -> PuzzleClassifier:
    return PuzzleClassifier(
        dataset_path=DATASET_PATH,
        embedding_service=FakeEmbeddingService(),
    )


def test_carrega_csv_e_colunas_obrigatorias() -> None:
    dataset = pd.read_csv(DATASET_PATH)
    assert {"frase", "token"}.issubset(dataset.columns)
    assert len(dataset) == 90


def test_classificacao_retorna_campos_esperados() -> None:
    resultado = criar_classificador().classificar("O que significam os símbolos da porta?")

    assert {
        "pergunta",
        "token",
        "similaridade",
        "frase_referencia",
        "top_resultados",
    }.issubset(resultado)
    assert resultado["token"] == "TOKEN_SIMBOLOS"
    assert isinstance(resultado["frase_referencia"], str)


def test_top_k_tem_tres_resultados() -> None:
    resultado = criar_classificador().classificar("O relógio mostra uma pista?")

    assert len(resultado["top_resultados"]) == 3
    similaridades = [item["similaridade"] for item in resultado["top_resultados"]]
    assert similaridades == sorted(similaridades, reverse=True)


def test_similaridade_esta_em_intervalo_valido() -> None:
    resultado = criar_classificador().classificar("Preciso contar as figuras do mural?")

    assert -1.0 <= resultado["similaridade"] <= 1.0


def test_pergunta_vazia_e_rejeitada() -> None:
    classificador = criar_classificador()

    with pytest.raises(ValueError, match="não pode ser vazia"):
        classificador.classificar("   ")


def test_obter_dica_retorna_texto_cadastrado() -> None:
    dica = obter_dica("TOKEN_MURAL")

    assert dica
    assert "mural" in dica.lower()


def test_token_desconhecido_nao_causa_excecao() -> None:
    dica = obter_dica("TOKEN_INEXISTENTE")

    assert dica
    assert "não há" in dica.lower()
