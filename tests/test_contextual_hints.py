"""Testes da seleção determinística de subintenções e do contexto."""

from __future__ import annotations

from typing import Any

import pytest

from src.game import HintManager, PuzzleEngine, load_puzzle_config


class FakeClassifier:
    def __init__(self, token: str) -> None:
        self.token = token
        self.chamadas: list[tuple[str, dict[str, Any]]] = []

    def classificar(self, pergunta: str, **kwargs: Any) -> dict[str, Any]:
        self.chamadas.append((pergunta, kwargs))
        return {
            "pergunta": pergunta,
            "token": self.token,
            "token_intencao": self.token,
            "similaridade": 0.91,
            "threshold": kwargs.get("threshold"),
        }


def _perguntar(
    engine: PuzzleEngine,
    pergunta: str,
    token: str = "TOKEN_COFRE",
    classificador: FakeClassifier | None = None,
) -> str:
    antes = engine.obter_estado().to_dict()
    classificador = classificador or FakeClassifier(token)
    resultado = engine.processar_pergunta(pergunta, classificador)
    assert engine.obter_estado().to_dict() == antes
    return resultado["dica"]


def _engine_no_estado(nome: str) -> PuzzleEngine:
    engine = PuzzleEngine(load_puzzle_config())
    if nome in {"cofre", "fotografia", "mural", "porta"}:
        engine.abrir_cofre("091530")
    if nome in {"fotografia", "mural", "porta"}:
        engine.encontrar_fotografia()
    if nome in {"mural", "porta"}:
        engine.completar_mural()
    if nome == "porta":
        engine.abrir_porta("2534")
    return engine


ESTADOS = ("inicial", "cofre", "fotografia", "mural", "porta")


def test_cofre_diferencia_senha_conteudo_finalidade_e_abertura() -> None:
    engine = PuzzleEngine(load_puzzle_config())

    dica_senha = _perguntar(engine, "Como consigo a senha do cofre?")
    dica_conteudo = _perguntar(engine, "O que existe dentro do cofre?")
    dica_finalidade = _perguntar(engine, "Qual é a finalidade do cofre?")
    dica_abertura = _perguntar(engine, "Como faço a abertura do cofre?")

    assert len({dica_senha, dica_conteudo, dica_finalidade, dica_abertura}) == 4
    assert "relógio" in dica_senha.casefold()
    assert "fotografia" not in dica_conteudo.casefold()
    assert "guardar" in dica_finalidade.casefold()
    assert "painel" in dica_abertura.casefold()
    assert "091530" not in dica_senha


@pytest.mark.parametrize("estado", ESTADOS)
def test_pergunta_sobre_senha_do_cofre_preserva_assunto_em_todo_progresso(estado: str) -> None:
    dica = _perguntar(_engine_no_estado(estado), "Como consigo a senha do cofre?")
    assert "relógio" in dica.casefold()
    assert "combinação" in dica.casefold()
    assert "091530" not in dica


@pytest.mark.parametrize("estado", ESTADOS)
def test_pergunta_sobre_abertura_preserva_procedimento_em_todo_progresso(estado: str) -> None:
    dica = _perguntar(_engine_no_estado(estado), "Como faço para abrir o cofre?")
    texto = dica.casefold()
    assert "painel" in texto or "procedimento" in texto
    assert "091530" not in dica


@pytest.mark.parametrize("estado", ESTADOS)
def test_pergunta_sobre_conteudo_respeita_progressao_sem_mudar_assunto(estado: str) -> None:
    dica = _perguntar(_engine_no_estado(estado), "O que existe dentro do cofre?")
    texto = dica.casefold()
    if estado == "inicial":
        assert "primeiro" in texto
        assert "fotografia" not in texto
    else:
        assert "fotografia" in texto or "mural" in texto


@pytest.mark.parametrize("estado", ESTADOS)
def test_pergunta_sobre_finalidade_preserva_finalidade(estado: str) -> None:
    dica = _perguntar(_engine_no_estado(estado), "Para que serve o cofre?")
    assert "serve" in dica.casefold()
    assert "cofre" in dica.casefold()
    assert "091530" not in dica


def test_estado_e_classificacao_sao_preservados_ao_selecionar_dica() -> None:
    engine = PuzzleEngine(load_puzzle_config())
    classificador = FakeClassifier("TOKEN_COFRE")
    estado_antes = engine.obter_estado().to_dict()

    resultado = engine.processar_pergunta(
        "Como consigo a senha do cofre?",
        classificador,
        modo="rejeicao",
        threshold=0.50,
    )

    assert resultado["classificacao"]["token"] == "TOKEN_COFRE"
    assert resultado["classificacao"]["similaridade"] == 0.91
    assert classificador.chamadas == [("Como consigo a senha do cofre?", {
        "modo": "rejeicao",
        "threshold": 0.50,
    })]
    assert engine.obter_estado().to_dict() == estado_antes


def test_token_rejeitado_nao_e_redirecionado_por_palavras_da_pergunta() -> None:
    engine = PuzzleEngine(load_puzzle_config())
    resultado = engine.processar_pergunta(
        "Qual é a senha do cofre?",
        FakeClassifier("TOKEN_NAO_RELACIONADO"),
    )

    assert "não foi possível" in resultado["dica"].casefold()
    assert "relógio" not in resultado["dica"].casefold()


def test_outros_tokens_priorizam_subintencao_apos_o_progresso() -> None:
    manager = HintManager()
    estado_completo = {
        "cofre_aberto": True,
        "foto_encontrada": True,
        "mural_completo": True,
        "porta_aberta": True,
    }

    assert "combinação" in manager.selecionar_dica(
        "TOKEN_RELOGIO", "Como os ponteiros ajudam a descobrir a senha do cofre?", estado_completo
    ).casefold()
    assert "quantidade" in manager.selecionar_dica(
        "TOKEN_MURAL", "Como devo contar os animais?", estado_completo
    ).casefold()
    assert "enciclopédia" in manager.selecionar_dica(
        "TOKEN_LIVRO", "Como associar os animais aos símbolos?", estado_completo
    ).casefold()
    assert "sequência" in manager.selecionar_dica(
        "TOKEN_SIMBOLOS", "Qual é a ordem dos símbolos?", estado_completo
    ).casefold()


@pytest.mark.parametrize(
    ("token", "pergunta", "trechos"),
    [
        ("TOKEN_RELOGIO", "Qual horário indicam os ponteiros?", ("horas", "ponteiros")),
        ("TOKEN_RELOGIO", "Qual relógio fornece o código do cofre?", ("combinação", "cofre")),
        ("TOKEN_MURAL", "Quais animais estão representados?", ("animais", "mural")),
        ("TOKEN_MURAL", "Como devo contar os animais?", ("quantidade", "totais")),
        ("TOKEN_LIVRO", "O que contém a enciclopédia?", ("enciclopédia", "correspondência")),
        ("TOKEN_LIVRO", "Como traduzir os símbolos?", ("enciclopédia", "traduzir")),
        ("TOKEN_SIMBOLOS", "O que significam os símbolos?", ("significado", "símbolos")),
        ("TOKEN_SIMBOLOS", "Como uso a sequência na porta?", ("sequência", "porta")),
    ],
)
def test_avaliacao_funcional_automatizada(token: str, pergunta: str, trechos: tuple[str, str]) -> None:
    dica = HintManager().selecionar_dica(token, pergunta, {
        "cofre_aberto": True,
        "foto_encontrada": True,
        "mural_completo": True,
        "porta_aberta": False,
    }).casefold()
    assert all(trecho.casefold() in dica for trecho in trechos)


def test_dicas_customizadas_preservam_comportamento_legado() -> None:
    manager = HintManager({"TOKEN_COFRE": "Dica customizada"})
    assert manager.selecionar_dica("TOKEN_COFRE", "Como consigo a senha?", {}) == "Dica customizada"


def test_padroes_de_subintencao_respeitam_palavras_inteiras() -> None:
    dica = HintManager().selecionar_dica(
        "TOKEN_RELOGIO", "O objeto relojoeiro não foi identificado.",
        {"cofre_aberto": False},
    )
    assert "combinação" not in dica.casefold()


@pytest.mark.parametrize(
    ("pergunta", "token", "trechos"),
    [
        ("Como consigo a senha do cofre?", "TOKEN_COFRE", ("relógio",)),
        ("Como encontro a combinação do cofre?", "TOKEN_COFRE", ("relógio",)),
        ("Qual relógio fornece o código do cofre?", "TOKEN_RELOGIO", ("combinação", "cofre")),
        ("O que existe dentro do cofre?", "TOKEN_COFRE", ("cofre",)),
        ("Para que serve o cofre?", "TOKEN_COFRE", ("cofre", "serve")),
        ("Como faço para abrir o cofre?", "TOKEN_COFRE", ("painel",)),
    ],
)
@pytest.mark.parametrize("estado", ESTADOS)
def test_casos_de_regressao_do_cofre_em_todos_os_estados(
    pergunta: str, token: str, trechos: tuple[str, ...], estado: str
) -> None:
    dica = _perguntar(_engine_no_estado(estado), pergunta, token).casefold()
    assert all(trecho.casefold() in dica for trecho in trechos)
    assert "091530" not in dica


def test_mural_inicial_nao_revela_fotografia_ou_cofre() -> None:
    dica = _perguntar(
        PuzzleEngine(load_puzzle_config()),
        "Como completo a parte ausente do mural?",
        token="TOKEN_MURAL",
    ).casefold()

    assert "investigue" in dica or "investigue os objetos" in dica
    assert "fotografia" not in dica
    assert "cofre" not in dica


def test_contagem_inicial_orienta_parte_visivel_sem_revelar_fotografia() -> None:
    dica = _perguntar(
        PuzzleEngine(load_puzzle_config()),
        "Como devo contar os animais?",
        token="TOKEN_MURAL",
    ).casefold()

    assert "visível" in dica
    assert "informações ausentes" in dica
    assert "fotografia" not in dica
    assert "cofre" not in dica


def test_mural_com_cofre_aberto_orienta_examinar_pista_disponivel() -> None:
    dica = _perguntar(
        _engine_no_estado("cofre"),
        "Como encontro a parte que falta do mural?",
        token="TOKEN_MURAL",
    ).casefold()

    assert "cofre" in dica
    assert "examine" in dica


def test_mural_apos_fotografia_orienta_combinar_as_duas_partes() -> None:
    dica = _perguntar(
        _engine_no_estado("fotografia"),
        "Como completo o mural?",
        token="TOKEN_MURAL",
    ).casefold()

    assert "fotografia" in dica
    assert "totais" in dica


def test_mural_completo_orienta_numeros_para_enciclopedia_e_simbolos() -> None:
    dica = _perguntar(
        _engine_no_estado("mural"),
        "O que faço com os números dos animais?",
        token="TOKEN_MURAL",
    ).casefold()

    assert "totais" in dica
    assert "enciclopédia" in dica
    assert "fotografia" not in dica


def test_perguntas_sobre_fotografia_nao_revelam_localizacao_antes_do_cofre() -> None:
    dica = _perguntar(
        PuzzleEngine(load_puzzle_config()),
        "Onde encontro a fotografia?",
        token="TOKEN_MURAL",
    ).casefold()

    assert "informação ainda não está disponível" in dica
    assert "dentro do cofre" not in dica


def test_mensagens_de_acesso_antecipado_nao_revelam_localizacao() -> None:
    engine = PuzzleEngine(load_puzzle_config())

    examinar = engine.examinar_objeto("fotografia")
    encontrar = engine.encontrar_fotografia()

    assert "dentro do cofre" not in examinar["mensagem"].casefold()
    assert "cofre fechado" not in encontrar["mensagem"].casefold()
