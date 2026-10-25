"""Testes da integração token → dica sem carregar o modelo real."""

from __future__ import annotations

from src.game import HintManager, PuzzleEngine, load_puzzle_config


class FakeClassifier:
    def __init__(self, token: str) -> None:
        self.token = token
        self.chamadas = 0

    def classificar(self, pergunta: str) -> dict[str, object]:
        self.chamadas += 1
        return {"pergunta": pergunta, "token": self.token, "similaridade": 0.9}


def test_tokens_retorna_dicas_correspondentes() -> None:
    manager = HintManager()
    esperadas = {
        "TOKEN_SIMBOLOS": "ordem dos símbolos",
        "TOKEN_LIVRO": "enciclopédia",
        "TOKEN_MURAL": "quantidade",
        "TOKEN_COFRE": "cofre",
        "TOKEN_RELOGIO": "horas",
        "TOKEN_NAO_RELACIONADO": "Não foi possível",
    }
    for token, trecho in esperadas.items():
        assert trecho.casefold() in manager.obter_dica(token).casefold()


def test_token_desconhecido_e_tratado() -> None:
    assert "não há" in HintManager().obter_dica("TOKEN_INEXISTENTE").casefold()


def test_integracao_preserva_classificacao_e_nao_muda_estado() -> None:
    engine = PuzzleEngine(load_puzzle_config())
    classificador = FakeClassifier("TOKEN_MURAL")
    antes = engine.obter_estado().to_dict()
    resultado = engine.processar_pergunta("Como conto os animais?", classificador)
    assert resultado["token"] == "TOKEN_MURAL"
    assert resultado["dica"]
    assert resultado["classificacao"]["pergunta"] == "Como conto os animais?"
    assert classificador.chamadas == 1
    assert engine.obter_estado().to_dict() == antes


def test_pergunta_nao_relacionada_usa_token_de_fallback() -> None:
    engine = PuzzleEngine(load_puzzle_config())
    resultado = engine.processar_pergunta("Qual a capital do país?", FakeClassifier("TOKEN_NAO_RELACIONADO"))
    assert "não foi possível" in resultado["dica"].casefold()
