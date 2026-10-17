"""Testes do baseline por palavras-chave."""

from __future__ import annotations

import pytest

from src.config import TOKEN_NAO_RELACIONADO
from src.keyword_classifier import (
    KeywordClassifier,
    normalizar,
    remover_acentos,
)


@pytest.fixture
def baseline() -> KeywordClassifier:
    return KeywordClassifier()


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------
def test_remover_acentos_preserva_letras_base() -> None:
    assert remover_acentos("relógio enciclopédia símbolo libélula") == (
        "relogio enciclopedia simbolo libelula"
    )


def test_normaliza_maiusculas_e_minusculas() -> None:
    assert normalizar("OS SÍMBOLOS DA PAREDE") == ["os", "simbolos", "da", "parede"]


def test_normaliza_remove_pontuacao() -> None:
    assert normalizar("Relógio, ponteiros... parados?!") == [
        "relogio",
        "ponteiros",
        "parados",
    ]


def test_normaliza_preserva_numeros() -> None:
    assert normalizar("O painel tem 4 digitos") == ["o", "painel", "tem", "4", "digitos"]


def test_normalizar_rejeita_valor_nao_textual() -> None:
    with pytest.raises(ValueError, match="precisa ser uma string"):
        normalizar(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contagem de correspondências
# ---------------------------------------------------------------------------
def test_reconhece_palavra_completa(baseline: KeywordClassifier) -> None:
    contagens = baseline.contar_correspondencias("Onde está o relógio?")

    assert contagens["TOKEN_RELOGIO"] == 1


def test_nao_casa_parte_de_palavra(baseline: KeywordClassifier) -> None:
    contagens = baseline.contar_correspondencias("Procuro um relojoeiro relogioeiro")

    assert contagens["TOKEN_RELOGIO"] == 0


def test_conta_palavras_chave_distintas(baseline: KeywordClassifier) -> None:
    contagens = baseline.contar_correspondencias(
        "Os ponteiros do relógio marcam horas, minutos e segundos"
    )

    # relogio, ponteiros, horas, minutos, segundos
    assert contagens["TOKEN_RELOGIO"] == 5


def test_repeticao_do_mesmo_termo_conta_uma_vez(baseline: KeywordClassifier) -> None:
    uma = baseline.contar_correspondencias("mural")
    tres = baseline.contar_correspondencias("mural mural mural")

    assert uma["TOKEN_MURAL"] == tres["TOKEN_MURAL"] == 1


def test_ausencia_de_correspondencias(baseline: KeywordClassifier) -> None:
    contagens = baseline.contar_correspondencias("Qual é a capital da Argentina?")

    assert set(contagens.values()) == {0}


# ---------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------
def test_classifica_pelo_maior_numero_de_correspondencias(
    baseline: KeywordClassifier,
) -> None:
    resultado = baseline.classificar(
        "Preciso contar a quantidade de figuras do mural?"
    )

    assert resultado["token"] == "TOKEN_MURAL"
    assert resultado["correspondencias"] >= 3
    assert resultado["empate"] is False


def test_sem_correspondencia_retorna_token_nao_relacionado(
    baseline: KeywordClassifier,
) -> None:
    resultado = baseline.classificar("Você conhece uma boa receita de lasanha?")

    assert resultado["token"] == TOKEN_NAO_RELACIONADO
    assert resultado["correspondencias"] == 0
    assert resultado["tokens_empatados"] == ()


def test_similaridade_e_sempre_nula(baseline: KeywordClassifier) -> None:
    resultado = baseline.classificar("Onde está o cofre?")

    assert resultado["similaridade"] is None


def test_empate_segue_ordem_documentada(baseline: KeywordClassifier) -> None:
    # "relogio" (TOKEN_RELOGIO) e "cofre" (TOKEN_COFRE) empatam com 1 ponto.
    resultado = baseline.classificar("O relógio ajuda com o cofre?")

    assert resultado["empate"] is True
    assert resultado["tokens_empatados"] == ("TOKEN_RELOGIO", "TOKEN_COFRE")
    assert resultado["token"] == "TOKEN_RELOGIO"
    assert baseline.ordem_desempate.index("TOKEN_RELOGIO") < baseline.ordem_desempate.index(
        "TOKEN_COFRE"
    )


def test_empate_e_deterministico(baseline: KeywordClassifier) -> None:
    pergunta = "O livro e o mural ajudam?"

    tokens = {baseline.classificar(pergunta)["token"] for _ in range(5)}

    assert len(tokens) == 1


def test_pergunta_vazia_e_rejeitada(baseline: KeywordClassifier) -> None:
    with pytest.raises(ValueError, match="não pode ser vazia"):
        baseline.classificar("   ")


def test_pergunta_nao_textual_e_rejeitada(baseline: KeywordClassifier) -> None:
    with pytest.raises(ValueError, match="não pode ser vazia"):
        baseline.classificar(None)  # type: ignore[arg-type]


def test_classificar_em_lote_preserva_a_ordem(baseline: KeywordClassifier) -> None:
    perguntas = ["Onde está o cofre?", "Qual a capital do Peru?", "E o mural?"]

    resultados = baseline.classificar_em_lote(perguntas)

    assert [r["pergunta"] for r in resultados] == perguntas
    assert [r["token"] for r in resultados] == [
        "TOKEN_COFRE",
        TOKEN_NAO_RELACIONADO,
        "TOKEN_MURAL",
    ]


# ---------------------------------------------------------------------------
# Configuração do vocabulário
# ---------------------------------------------------------------------------
def test_vocabulario_nao_usa_palavras_genericas_criticas(
    baseline: KeywordClassifier,
) -> None:
    genericas = {("senha",), ("codigo",), ("combinacao",), ("numero",), ("animal",), ("pista",)}

    for termos in baseline.palavras_chave.values():
        assert not genericas & set(termos)


def test_vocabulario_vazio_gera_erro() -> None:
    with pytest.raises(ValueError, match="ao menos um token"):
        KeywordClassifier(palavras_chave={})


def test_token_sem_palavras_gera_erro() -> None:
    with pytest.raises(ValueError, match="não possui palavras-chave"):
        KeywordClassifier(palavras_chave={"TOKEN_X": []})


def test_vocabulario_personalizado_com_expressao_de_duas_palavras() -> None:
    classificador = KeywordClassifier(
        palavras_chave={"TOKEN_SIMBOLOS": ["painel numerico"]},
        ordem_desempate=("TOKEN_SIMBOLOS",),
    )

    assert classificador.classificar("Uso o painel numérico?")["token"] == "TOKEN_SIMBOLOS"
    assert (
        classificador.classificar("O painel fica numerado?")["token"]
        == TOKEN_NAO_RELACIONADO
    )


def test_tokens_extras_entram_na_ordem_de_desempate() -> None:
    classificador = KeywordClassifier(
        palavras_chave={"TOKEN_ZZZ": ["zebra"], "TOKEN_RELOGIO": ["relogio"]},
        ordem_desempate=("TOKEN_RELOGIO",),
    )

    assert classificador.ordem_desempate == ("TOKEN_RELOGIO", "TOKEN_ZZZ")
    assert classificador.classificar("relogio zebra")["token"] == "TOKEN_RELOGIO"
