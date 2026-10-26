"""Testes de regressão dos artefatos derivados da Semana 5."""

from pathlib import Path

import pandas as pd

from scripts.consolidar_resultados_semana5 import (
    consolidar_comparacao,
    consolidar_erros,
    consolidar_funcional,
)


ROOT = Path(__file__).resolve().parents[1]


def test_consolidacao_preserva_metricas_historicas(tmp_path):
    tabela, _ = consolidar_comparacao(ROOT / "results", tmp_path)
    assert list(tabela["acertos"]) == [34, 40, 46]
    assert list(tabela["total"]) == [60, 60, 60]
    assert (tmp_path / "comparacao_final.md").exists()


def test_consolidacao_mantem_os_14_erros(tmp_path):
    erros = consolidar_erros(ROOT / "results", tmp_path)
    assert len(erros) == 14
    assert set(["pergunta", "token_esperado", "token_previsto", "similaridade", "categoria_erro"]).issubset(erros.columns)


def test_avaliacao_funcional_preserva_27_cenarios(tmp_path):
    cenarios, metricas, _ = consolidar_funcional(ROOT, tmp_path)
    assert cenarios is not None
    assert metricas is not None
    assert len(cenarios) == 27
    assert cenarios["id_avaliacao"].is_unique
    assert set(cenarios["id_pergunta"]) == set(range(1, 16))
    assert (tmp_path / "avaliacao_dicas_por_cenario.csv").exists()
    assert (tmp_path / "avaliacao_dicas_metricas.csv").exists()
