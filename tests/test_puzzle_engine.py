"""Testes das regras do puzzle sem depender do modelo de embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PUZZLE_CONFIG_PATH
from src.game import PuzzleConfig, PuzzleConfigError, PuzzleEngine, PuzzleState, load_puzzle_config


@pytest.fixture
def config() -> PuzzleConfig:
    return load_puzzle_config(PUZZLE_CONFIG_PATH)


@pytest.fixture
def engine(config: PuzzleConfig) -> PuzzleEngine:
    return PuzzleEngine(config)


def test_configuracao_e_estado_inicial(config: PuzzleConfig, engine: PuzzleEngine) -> None:
    assert config.horario == "09:15:30"
    assert config.senha_cofre == "091530"
    assert engine.obter_estado().to_dict() == {
        "cofre_aberto": False,
        "foto_encontrada": False,
        "mural_completo": False,
        "porta_aberta": False,
        "venceu": False,
    }


def test_configuracao_invalida_e_rejeitada(tmp_path: Path, config: PuzzleConfig) -> None:
    dados = {
        "nome": "Sala",
        "relogio": {"horario": "25:00:00"},
        "mural": {"animais": {"coruja": {"visivel": 1, "fotografia": 1}}},
        "enciclopedia": {"◆": "inexistente"},
        "parede": {"sequencia": ["◆"]},
    }
    caminho = tmp_path / "invalido.json"
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    with pytest.raises(PuzzleConfigError, match="horário"):
        PuzzleConfig.from_file(caminho)


def test_configuracao_valida_simbolos_e_totais(config: PuzzleConfig) -> None:
    assert set(config.sequencia) <= set(config.enciclopedia)
    assert config.totais_animais == {
        "libelula": 5,
        "coruja": 2,
        "tartaruga": 4,
        "raposa": 3,
    }
    assert config.senha_porta == "2534"


def test_relogio_e_examinacao_nao_alteram_estado(engine: PuzzleEngine) -> None:
    antes = engine.obter_estado().to_dict()
    resultado = engine.examinar_objeto("Relógio")
    assert resultado["sucesso"]
    assert "09:15:30" in resultado["mensagem"]
    assert engine.obter_estado().to_dict() == antes


def test_senhas_preservam_zeros_e_cofre_so_abre_com_senha_derivada(engine: PuzzleEngine) -> None:
    assert not engine.abrir_cofre("91530")["sucesso"]
    assert not engine.obter_estado().cofre_aberto
    assert engine.abrir_cofre("091530")["sucesso"]
    assert engine.obter_estado().cofre_aberto
    assert engine.abrir_cofre("000000")["evento"] == "COFRE_JA_ABERTO"


def test_fotografia_depende_do_cofre_e_e_idempotente(engine: PuzzleEngine) -> None:
    assert not engine.encontrar_fotografia()["sucesso"]
    engine.abrir_cofre("091530")
    assert engine.encontrar_fotografia()["sucesso"]
    assert engine.encontrar_fotografia()["evento"] == "FOTOGRAFIA_JA_ENCONTRADA"
    assert engine.obter_estado().foto_encontrada


def test_mural_depende_da_fotografia_e_calcula_totais(engine: PuzzleEngine) -> None:
    bloqueado = engine.completar_mural()
    assert not bloqueado["sucesso"]
    engine.abrir_cofre("091530")
    engine.encontrar_fotografia()
    completo = engine.completar_mural()
    assert completo["sucesso"]
    assert completo["dados"]["totais"]["libelula"] == 5
    assert engine.obter_estado().mural_completo
    assert engine.completar_mural()["evento"] == "MURAL_JA_COMPLETO"


def test_enciclopedia_nao_revela_valores_e_parede_preserva_sequencia(engine: PuzzleEngine) -> None:
    livro = engine.consultar_enciclopedia()
    assert livro["dados"]["associacoes"]["◆"] == "Libélula"
    assert "5" not in livro["mensagem"]
    parede = engine.examinar_objeto("parede")
    assert parede["dados"]["sequencia"] == ["☼", "◆", "✦", "△"]
    assert "2534" not in parede["mensagem"]


def test_porta_usa_senha_derivada_e_registra_vitoria(engine: PuzzleEngine) -> None:
    assert engine.calcular_senha_porta() == "2534"
    assert not engine.abrir_porta("0000")["sucesso"]
    vitoria = engine.abrir_porta("2534")
    assert vitoria["sucesso"]
    assert vitoria["evento"] == "VITORIA"
    assert engine.obter_estado().venceu
    assert engine.abrir_porta("0000")["evento"] == "VITORIA_JA_REGISTRADA"


def test_reinicio_restabelece_estado(engine: PuzzleEngine) -> None:
    engine.abrir_cofre("091530")
    engine.encontrar_fotografia()
    engine.completar_mural()
    engine.abrir_porta("2534")
    assert engine.reiniciar()["sucesso"]
    assert engine.obter_estado() == PuzzleState()


def test_instancias_nao_compartilham_estado(config: PuzzleConfig) -> None:
    primeiro = PuzzleEngine(config)
    segundo = PuzzleEngine(config)
    primeiro.abrir_cofre("091530")
    assert primeiro.obter_estado().cofre_aberto
    assert not segundo.obter_estado().cofre_aberto


def test_configuracao_rejeita_quantidade_negativa_e_simbolo_desconhecido() -> None:
    dados = {
        "nome": "Sala",
        "relogio": {"horario": "09:15:30"},
        "mural": {"animais": {"coruja": {"visivel": -1, "fotografia": 1}}},
        "enciclopedia": {"◆": "coruja"},
        "parede": {"sequencia": ["☼"]},
    }
    with pytest.raises(PuzzleConfigError, match="não negativo"):
        PuzzleConfig.from_dict(dados)

    dados["mural"]["animais"]["coruja"]["visivel"] = 1
    dados["parede"]["sequencia"] = ["☼"]
    with pytest.raises(PuzzleConfigError, match="não existe"):
        PuzzleConfig.from_dict(dados)
