"""Testes de regressão dos contratos de integridade da Semana 5."""

from pathlib import Path

from scripts.validar_integridade_semana5 import executar_validacao, normalizar_frase


ROOT = Path(__file__).resolve().parents[1]


def test_normalizacao_eh_apenas_para_comparacao():
    assert normalizar_frase("  Símbolos  da\tPAREDE ") == "simbolos da parede"


def test_integridade_datasets_reais(tmp_path):
    resultado = executar_validacao(ROOT, tmp_path)
    assert resultado["integridade_aprovada"] is True
    assert resultado["referencias"]["total"] == 90
    assert resultado["avaliacao"]["total"] == 100
    assert resultado["particoes"]["total_validacao"] == 40
    assert resultado["particoes"]["total_teste"] == 60
    assert resultado["frases_idênticas_referencia_avaliacao"] == []


def test_integridade_nao_altera_datasets(tmp_path):
    referencia = (ROOT / "data" / "frases_referencia.csv").read_bytes()
    avaliacao = (ROOT / "data" / "teste_manual.csv").read_bytes()
    executar_validacao(ROOT, tmp_path)
    assert (ROOT / "data" / "frases_referencia.csv").read_bytes() == referencia
    assert (ROOT / "data" / "teste_manual.csv").read_bytes() == avaliacao
