"""Testes da camada de integração usada pela interface Streamlit.

Os testes usam a engine e classificadores falsos para não baixar o modelo real.
O teste de fumaça executa o script pelo AppTest quando a API de testes do
Streamlit está disponível.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import streamlit_app
from src.config import MODO_REJEICAO
from src.game import PuzzleEngine


class FakeClassifier:
    def __init__(self, token: str = "TOKEN_MURAL") -> None:
        self.token = token
        self.chamadas = 0
        self.argumentos = []

    def classificar(self, pergunta: str, **kwargs):
        self.chamadas += 1
        self.argumentos.append(kwargs)
        return {
            "pergunta": pergunta,
            "token": self.token,
            "similaridade": 0.91,
            "frase_referencia": "frase de teste",
            "top_resultados": [],
            "modo": kwargs.get("modo"),
            "threshold": kwargs.get("threshold"),
            "rejeitada": self.token == "TOKEN_NAO_RELACIONADO",
        }


def test_app_inicializa_engine_e_preserva_a_mesma_instancia() -> None:
    session = {}
    primeira = streamlit_app.obter_engine(session)
    segunda = streamlit_app.obter_engine(session)

    assert isinstance(primeira, PuzzleEngine)
    assert primeira is segunda
    assert streamlit_app.obter_estado_apresentacao(primeira)["venceu"] is False


def test_sessoes_distintas_nao_compartilham_estado() -> None:
    primeira = streamlit_app.obter_engine({})
    segunda = streamlit_app.obter_engine({})

    primeira.abrir_cofre("091530")

    assert primeira.obter_estado().cofre_aberto is True
    assert segunda.obter_estado().cofre_aberto is False


def test_fluxo_dos_objetos_preserva_senhas_como_texto() -> None:
    session = {}
    engine = streamlit_app.obter_engine(session)

    incorreta = engine.abrir_cofre("91530")
    assert incorreta["sucesso"] is False
    assert engine.obter_estado().cofre_aberto is False

    correta = engine.abrir_cofre("091530")
    assert correta["sucesso"] is True
    assert engine.encontrar_fotografia()["sucesso"] is True
    mural = engine.completar_mural()
    assert mural["dados"]["totais"] == {
        "libelula": 5,
        "coruja": 2,
        "tartaruga": 4,
        "raposa": 3,
    }

    enciclopedia = streamlit_app.executar_acao_objeto(engine, session, "enciclopedia")
    simbolos = streamlit_app.executar_acao_objeto(engine, session, "simbolos")
    assert "5" not in enciclopedia["mensagem"]
    assert simbolos["dados"]["sequencia"] == ["☼", "◆", "✦", "△"]

    assert engine.abrir_porta("0000")["sucesso"] is False
    assert engine.abrir_porta("2534")["evento"] == "VITORIA"


def test_pergunta_usa_rejeicao_threshold_e_nao_altera_estado() -> None:
    engine = streamlit_app.criar_engine()
    classificador = FakeClassifier()
    antes = engine.obter_estado().to_dict()

    resultado = streamlit_app.executar_pergunta(engine, "Como conto os animais?", classificador)

    assert resultado["dica"]
    assert classificador.chamadas == 1
    assert classificador.argumentos == [{"modo": MODO_REJEICAO, "threshold": 0.50}]
    assert engine.obter_estado().to_dict() == antes


def test_pergunta_nao_relacionada_preserva_mensagem_de_rejeicao() -> None:
    resultado = streamlit_app.executar_pergunta(
        streamlit_app.criar_engine(),
        "Qual a capital do país?",
        FakeClassifier("TOKEN_NAO_RELACIONADO"),
    )
    assert "não foi possível" in resultado["dica"].casefold()


def test_pergunta_vazia_e_rejeitada_antes_do_classificador() -> None:
    with pytest.raises(ValueError, match="não pode ser vazia"):
        streamlit_app.executar_pergunta(streamlit_app.criar_engine(), "  ", FakeClassifier())


def test_reinicio_limpa_estado_e_mensagens_da_partida(monkeypatch) -> None:
    session = {
        streamlit_app.OBJECT_RESULTS_KEY: {"relogio": {"sucesso": True}},
        streamlit_app.LAST_HINT_KEY: {"dica": "dica"},
        streamlit_app.CLASSIFIER_ERROR_KEY: "erro anterior",
        "cofre_senha": "091530",
        "porta_senha": "2534",
        "pergunta": "pergunta anterior",
        streamlit_app.VICTORY_ANIMATED_KEY: True,
    }
    engine = streamlit_app.obter_engine(session)
    engine.abrir_cofre("091530")

    resultado = streamlit_app.executar_reinicio(engine, session)

    assert resultado["evento"] == "JOGO_REINICIADO"
    assert engine.obter_estado().to_dict() == {
        "cofre_aberto": False,
        "foto_encontrada": False,
        "mural_completo": False,
        "porta_aberta": False,
        "venceu": False,
    }
    assert session[streamlit_app.OBJECT_RESULTS_KEY] == {}
    assert streamlit_app.LAST_HINT_KEY not in session
    assert streamlit_app.CLASSIFIER_ERROR_KEY not in session
    assert session["cofre_senha"] == ""
    assert session["porta_senha"] == ""
    assert session["pergunta"] == ""
    assert session[streamlit_app.VICTORY_ANIMATED_KEY] is False


def test_vitoria_animada_apenas_uma_vez_e_reinicia(monkeypatch) -> None:
    session = {}
    chamadas = []
    mensagens = []
    monkeypatch.setattr(streamlit_app.st, "balloons", lambda: chamadas.append("balloons"))
    monkeypatch.setattr(streamlit_app.st, "success", lambda mensagem: mensagens.append(mensagem))

    streamlit_app._mostrar_vitoria(session, True)
    streamlit_app._mostrar_vitoria(session, True)

    assert chamadas == ["balloons"]
    assert len(mensagens) == 2
    assert session[streamlit_app.VICTORY_ANIMATED_KEY] is True

    streamlit_app.executar_reinicio(streamlit_app.criar_engine(), session)
    streamlit_app._mostrar_vitoria(session, True)
    assert chamadas == ["balloons", "balloons"]


def test_classificador_cacheado_nao_e_instanciado_a_cada_chamada(monkeypatch) -> None:
    chamadas = []

    class FakePuzzleClassifier:
        def __init__(self, dataset_path):
            chamadas.append(Path(dataset_path))

    monkeypatch.setattr(streamlit_app, "PuzzleClassifier", FakePuzzleClassifier)
    streamlit_app.carregar_classificador.clear()
    try:
        primeiro = streamlit_app.carregar_classificador("data/frases_referencia.csv")
        segundo = streamlit_app.carregar_classificador("data/frases_referencia.csv")
    finally:
        streamlit_app.carregar_classificador.clear()

    assert primeiro is segundo
    assert chamadas == [Path("data/frases_referencia.csv")]


def test_streamlit_smoke_apresenta_sala_e_objetos() -> None:
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.fail("A API streamlit.testing.v1 é necessária para o smoke test.")

    app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py"), default_timeout=10)
    app.run()

    assert not app.exception
    assert app.title[0].value == "✦ Sala dos Símbolos"
    texto = " ".join(element.value for element in app.markdown)
    assert "Relógio" in texto
    assert "Cofre" in texto
    assert "Fotografia" not in texto
    assert "Mural" in texto
    assert "Enciclopédia" in texto
    assert "Símbolos da parede" in texto
    assert "Porta" in texto


def test_resultado_de_acao_e_preservado_para_o_rerun() -> None:
    session = {}
    engine = streamlit_app.criar_engine()

    resultado = engine.abrir_cofre("091530")
    streamlit_app.registrar_resultado(session, resultado, objeto="cofre")

    assert session[streamlit_app.LAST_ACTION_KEY] is resultado
    assert session[streamlit_app.OBJECT_RESULTS_KEY]["cofre"] is resultado
    assert resultado["mensagem"] == "O cofre foi aberto."


def test_fotografia_so_e_renderizada_apos_abertura_do_cofre(monkeypatch) -> None:
    session = {}
    engine = streamlit_app.criar_engine()
    renderizados = []

    monkeypatch.setattr(streamlit_app.st, "markdown", lambda texto: renderizados.append(texto))
    monkeypatch.setattr(streamlit_app.st, "write", lambda texto: renderizados.append(texto))
    monkeypatch.setattr(streamlit_app.st, "button", lambda *args, **kwargs: False)

    streamlit_app._renderizar_fotografia(engine, session)
    assert renderizados == []

    engine.abrir_cofre("091530")
    streamlit_app._renderizar_fotografia(engine, session)

    assert any("Fotografia" in str(texto) for texto in renderizados)
    assert any("indisponível" in str(texto) for texto in renderizados)


def test_estado_lateral_omite_fotografia_antes_de_abrir_cofre(monkeypatch) -> None:
    engine = streamlit_app.criar_engine()
    exibidos = []

    monkeypatch.setattr(streamlit_app.st.sidebar, "subheader", lambda texto: exibidos.append(texto))
    monkeypatch.setattr(streamlit_app.st.sidebar, "write", lambda texto: exibidos.append(texto))
    monkeypatch.setattr(streamlit_app.st.sidebar, "success", lambda texto: exibidos.append(texto))

    streamlit_app._mostrar_estado(engine)
    assert not any("Fotografia" in str(texto) for texto in exibidos)

    engine.abrir_cofre("091530")
    exibidos.clear()
    streamlit_app._mostrar_estado(engine)
    assert any("Fotografia" in str(texto) for texto in exibidos)


def test_solicitar_dica_preserva_as_quatro_flags_do_estado() -> None:
    engine = streamlit_app.criar_engine()
    engine.abrir_cofre("091530")
    engine.encontrar_fotografia()
    engine.completar_mural()
    engine.abrir_porta("2534")
    antes = engine.obter_estado().to_dict()

    resultado = streamlit_app.executar_pergunta(
        engine,
        "O que faço com os números dos animais?",
        FakeClassifier("TOKEN_MURAL"),
    )

    assert resultado["dica"]
    assert engine.obter_estado().to_dict() == antes
