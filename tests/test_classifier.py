"""Testes do classificador de embeddings, sem carregar o modelo real."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classifier import PuzzleClassifier
from src.config import MODO_PADRAO, MODO_REJEICAO, TOKEN_NAO_RELACIONADO
from src.responses import obter_dica


# ---------------------------------------------------------------------------
# Carregamento e validação do dataset
# ---------------------------------------------------------------------------
def test_carrega_csv_e_colunas_obrigatorias(caminho_referencia: Path) -> None:
    dataset = pd.read_csv(caminho_referencia)

    assert {"frase", "token"}.issubset(dataset.columns)
    assert len(dataset) == 90


def test_referencia_tem_quinze_frases_por_token(caminho_referencia: Path) -> None:
    dataset = pd.read_csv(caminho_referencia)

    contagem = dataset["token"].value_counts()
    assert set(contagem.index) == {
        "TOKEN_SIMBOLOS",
        "TOKEN_LIVRO",
        "TOKEN_MURAL",
        "TOKEN_COFRE",
        "TOKEN_RELOGIO",
        "TOKEN_NAO_RELACIONADO",
    }
    assert contagem.unique().tolist() == [15]


def test_dataset_inexistente_gera_erro(tmp_path: Path, fake_embeddings) -> None:
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        PuzzleClassifier(tmp_path / "ausente.csv", embedding_service=fake_embeddings)


def test_dataset_vazio_gera_erro(tmp_path: Path, fake_embeddings) -> None:
    caminho = tmp_path / "vazio.csv"
    caminho.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="vazio"):
        PuzzleClassifier(caminho, embedding_service=fake_embeddings)


def test_dataset_sem_registros_gera_erro(tmp_path: Path, fake_embeddings) -> None:
    caminho = tmp_path / "sem_registros.csv"
    caminho.write_text("frase,token\n", encoding="utf-8")

    with pytest.raises(ValueError, match="registros"):
        PuzzleClassifier(caminho, embedding_service=fake_embeddings)


def test_colunas_ausentes_geram_erro(tmp_path: Path, fake_embeddings) -> None:
    caminho = tmp_path / "colunas.csv"
    caminho.write_text("pergunta,classe\nalgo,TOKEN_LIVRO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        PuzzleClassifier(caminho, embedding_service=fake_embeddings)


def test_frase_vazia_no_dataset_gera_erro(tmp_path: Path, fake_embeddings) -> None:
    caminho = tmp_path / "frase_vazia.csv"
    caminho.write_text('frase,token\n"   ",TOKEN_LIVRO\n', encoding="utf-8")

    with pytest.raises(ValueError, match="textos vazios"):
        PuzzleClassifier(caminho, embedding_service=fake_embeddings)


def test_top_k_invalido_gera_erro(caminho_referencia: Path, fake_embeddings) -> None:
    with pytest.raises(ValueError, match="maior que zero"):
        PuzzleClassifier(
            caminho_referencia, embedding_service=fake_embeddings, top_k=0
        )


def test_threshold_fora_do_intervalo_gera_erro(
    caminho_referencia: Path, fake_embeddings
) -> None:
    with pytest.raises(ValueError, match="entre -1 e 1"):
        PuzzleClassifier(
            caminho_referencia,
            embedding_service=fake_embeddings,
            similarity_threshold=1.5,
        )


# ---------------------------------------------------------------------------
# Estrutura do resultado
# ---------------------------------------------------------------------------
def test_classificacao_retorna_campos_esperados(classificador: PuzzleClassifier) -> None:
    resultado = classificador.classificar("O que significam os símbolos da porta?")

    assert {
        "pergunta",
        "token",
        "similaridade",
        "frase_referencia",
        "top_resultados",
        "token_intencao",
        "modo",
        "threshold",
        "rejeitada",
    }.issubset(resultado)
    assert resultado["token"] == "TOKEN_SIMBOLOS"
    assert isinstance(resultado["frase_referencia"], str)
    assert resultado["modo"] == MODO_PADRAO


def test_top_k_tem_tres_resultados_ordenados(classificador: PuzzleClassifier) -> None:
    resultado = classificador.classificar("O relógio mostra uma pista?")

    assert len(resultado["top_resultados"]) == 3
    similaridades = [item["similaridade"] for item in resultado["top_resultados"]]
    assert similaridades == sorted(similaridades, reverse=True)
    assert resultado["similaridade"] == similaridades[0]


def test_similaridade_esta_em_intervalo_valido(classificador: PuzzleClassifier) -> None:
    resultado = classificador.classificar("Preciso contar as figuras do mural?")

    assert -1.0 <= resultado["similaridade"] <= 1.0


def test_pergunta_vazia_e_rejeitada(classificador: PuzzleClassifier) -> None:
    with pytest.raises(ValueError, match="não pode ser vazia"):
        classificador.classificar("   ")


def test_pergunta_nao_textual_e_rejeitada(classificador: PuzzleClassifier) -> None:
    with pytest.raises(ValueError, match="não pode ser vazia"):
        classificador.classificar(None)  # type: ignore[arg-type]


def test_modo_invalido_gera_erro(classificador: PuzzleClassifier) -> None:
    with pytest.raises(ValueError, match="Modo inválido"):
        classificador.classificar("O relógio é uma pista?", modo="outro")


# ---------------------------------------------------------------------------
# Comportamento com e sem threshold
# ---------------------------------------------------------------------------
def test_sem_threshold_nunca_rejeita(classificador: PuzzleClassifier) -> None:
    resultado = classificador.classificar("Qual é a capital do Japão?")

    assert resultado["threshold"] is None
    assert resultado["rejeitada"] is False
    assert resultado["token"] == resultado["token_intencao"]


def test_threshold_alto_rejeita_no_modo_padrao(classificador: PuzzleClassifier) -> None:
    pergunta = "Preciso contar as figuras do mural e consultar o livro?"
    base = classificador.classificar(pergunta)
    assert base["similaridade"] < 1.0

    limiar = base["similaridade"] + 0.005
    resultado = classificador.classificar(pergunta, modo=MODO_PADRAO, threshold=limiar)

    assert resultado["rejeitada"] is True
    assert resultado["token"] == TOKEN_NAO_RELACIONADO
    assert resultado["token_intencao"] != TOKEN_NAO_RELACIONADO


def test_similaridade_igual_ao_limiar_e_aceita(classificador: PuzzleClassifier) -> None:
    # A regra é "aceita quando similaridade >= threshold".
    matriz = np.full((1, len(classificador.frases_referencia)), 0.50)

    resultado = classificador.classificar_de_matriz(
        ["uma pergunta qualquer"], matriz, MODO_REJEICAO, 0.50
    )[0]

    assert resultado["rejeitada"] is False
    assert resultado["token"] != TOKEN_NAO_RELACIONADO


def test_similaridade_abaixo_do_limiar_e_rejeitada(classificador: PuzzleClassifier) -> None:
    matriz = np.full((1, len(classificador.frases_referencia)), 0.30)

    resultado = classificador.classificar_de_matriz(
        ["uma pergunta qualquer"], matriz, MODO_REJEICAO, 0.50
    )[0]

    assert resultado["rejeitada"] is True
    assert resultado["token"] == TOKEN_NAO_RELACIONADO
    assert resultado["similaridade"] == pytest.approx(0.30)


def test_modo_rejeicao_exige_threshold(classificador: PuzzleClassifier) -> None:
    with pytest.raises(ValueError, match="exige um limiar"):
        classificador.classificar("O relógio é uma pista?", modo=MODO_REJEICAO)


def test_modo_rejeicao_ignora_referencias_nao_relacionadas(
    classificador: PuzzleClassifier,
) -> None:
    resultado = classificador.classificar(
        "Qual é a capital do Brasil e o maior planeta?",
        modo=MODO_REJEICAO,
        threshold=-1.0,
    )

    tokens_consultados = {item["token"] for item in resultado["top_resultados"]}
    assert TOKEN_NAO_RELACIONADO not in tokens_consultados
    assert resultado["token_intencao"] != TOKEN_NAO_RELACIONADO
    assert resultado["rejeitada"] is False


def test_modo_rejeicao_com_threshold_alto_devolve_nao_relacionado(
    classificador: PuzzleClassifier,
) -> None:
    resultado = classificador.classificar(
        "Como faço um bolo de chocolate?", modo=MODO_REJEICAO, threshold=0.99
    )

    assert resultado["token"] == TOKEN_NAO_RELACIONADO
    assert resultado["rejeitada"] is True


def test_indices_puzzle_excluem_token_nao_relacionado(
    classificador: PuzzleClassifier,
) -> None:
    tokens = {classificador.tokens_referencia[i] for i in classificador.indices_puzzle}

    assert TOKEN_NAO_RELACIONADO not in tokens
    assert len(classificador.indices_puzzle) == 75


def test_threshold_do_construtor_e_usado_por_padrao(
    caminho_referencia: Path, fake_embeddings
) -> None:
    classificador = PuzzleClassifier(
        caminho_referencia,
        embedding_service=fake_embeddings,
        similarity_threshold=0.99,
    )

    resultado = classificador.classificar(
        "Preciso contar as figuras do mural e consultar o livro?"
    )

    assert resultado["threshold"] == pytest.approx(0.99)
    assert resultado["similaridade"] < 0.99
    assert resultado["token"] == TOKEN_NAO_RELACIONADO


# ---------------------------------------------------------------------------
# Processamento em lote
# ---------------------------------------------------------------------------
def test_lote_equivale_a_chamadas_individuais(classificador: PuzzleClassifier) -> None:
    perguntas = [
        "O que significam os símbolos da parede?",
        "Preciso contar os animais do mural?",
        "Qual é a capital do Brasil?",
    ]

    em_lote = classificador.classificar_em_lote(perguntas)
    individuais = [classificador.classificar(pergunta) for pergunta in perguntas]

    assert [r["token"] for r in em_lote] == [r["token"] for r in individuais]
    assert [r["similaridade"] for r in em_lote] == pytest.approx(
        [r["similaridade"] for r in individuais]
    )


def test_matriz_similaridade_tem_formato_esperado(classificador: PuzzleClassifier) -> None:
    matriz = classificador.matriz_similaridade(["O relógio parou?", "E o cofre?"])

    assert matriz.shape == (2, 90)
    assert np.all(matriz >= -1.0) and np.all(matriz <= 1.0)


def test_similaridade_maxima_puzzle_usa_apenas_cinco_intencoes(
    classificador: PuzzleClassifier,
) -> None:
    matriz = classificador.matriz_similaridade(["Qual é a capital do Brasil?"])

    maximo_puzzle = classificador.similaridade_maxima_puzzle(matriz)[0]
    maximo_geral = matriz[0].max()

    assert maximo_puzzle <= maximo_geral + 1e-9
    assert maximo_puzzle == pytest.approx(
        matriz[0][classificador.indices_puzzle].max()
    )


def test_matriz_incompativel_gera_erro(classificador: PuzzleClassifier) -> None:
    with pytest.raises(ValueError, match="não corresponde"):
        classificador.classificar_de_matriz(["uma pergunta"], np.zeros((2, 90)))


def test_lote_vazio_gera_erro(classificador: PuzzleClassifier) -> None:
    with pytest.raises(ValueError, match="ao menos uma pergunta"):
        classificador.matriz_similaridade([])


# ---------------------------------------------------------------------------
# Dicas
# ---------------------------------------------------------------------------
def test_obter_dica_retorna_texto_cadastrado() -> None:
    dica = obter_dica("TOKEN_MURAL")

    assert dica
    assert "mural" in dica.lower()


def test_token_desconhecido_nao_causa_excecao() -> None:
    dica = obter_dica("TOKEN_INEXISTENTE")

    assert dica
    assert "não há" in dica.lower()
