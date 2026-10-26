"""Testes unitários da amostragem e rastreabilidade do benchmark da Semana 5."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import benchmark_latencia as benchmark

ROOT = Path(__file__).resolve().parents[1]


def _dataset_pequeno() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frase": [f"pergunta {i}" for i in range(12)],
            "token_esperado": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        }
    )


def test_amostra_estratificada_e_reprodutivel_sem_duplicatas():
    dataset = _dataset_pequeno()
    primeira = benchmark.amostrar_perguntas_estratificadas(dataset, por_classe=2, seed=42)
    segunda = benchmark.amostrar_perguntas_estratificadas(dataset, por_classe=2, seed=42)

    pd.testing.assert_frame_equal(primeira, segunda)
    assert len(primeira) == 6
    assert primeira["token_esperado"].value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}
    assert primeira["id_pergunta"].is_unique
    assert primeira["posicao_original"].is_unique
    assert set(primeira["semente"]) == {42}


def test_amostra_nao_depende_da_ordem_original():
    dataset = _dataset_pequeno()
    original = benchmark.amostrar_perguntas_estratificadas(dataset, por_classe=2, seed=42)
    embaralhado = dataset.sample(frac=1, random_state=99).reset_index(drop=True)
    alterada = benchmark.amostrar_perguntas_estratificadas(embaralhado, por_classe=2, seed=42)

    assert set(original["frase"]) == set(alterada["frase"])
    assert original["token_esperado"].value_counts().to_dict() == alterada["token_esperado"].value_counts().to_dict()


def test_amostra_informa_classe_insuficiente():
    dataset = pd.DataFrame({"frase": ["a", "b"], "token_esperado": ["A", "B"]})

    with pytest.raises(ValueError, match="possui 1 exemplos"):
        benchmark.amostrar_perguntas_estratificadas(dataset, por_classe=2, seed=42)


def test_resumo_calcula_estatisticas_dos_registros_da_mesma_execucao():
    registros = [
        {
            "id_execucao": "benchmark_teste",
            "configuracao": "palavras_chave",
            "fase": "subsequente",
            "tempo_ms": valor,
            "threshold": None,
        }
        for valor in (1.0, 2.0, 3.0)
    ]
    resumo = benchmark.resumo_tempos(registros)

    linha = resumo.iloc[0]
    assert linha["id_execucao"] == "benchmark_teste"
    assert linha["n_execucoes"] == 3
    assert linha["media_ms"] == pytest.approx(2.0)
    assert linha["mediana_ms"] == pytest.approx(2.0)
    assert linha["minimo_ms"] == pytest.approx(1.0)
    assert linha["maximo_ms"] == pytest.approx(3.0)
    assert linha["p95_ms"] == pytest.approx(np.percentile([1.0, 2.0, 3.0], 95))


def test_benchmark_mockado_gera_artefatos_associados(monkeypatch, tmp_path):
    class FakeEmbeddingService:
        def __init__(self, model_name):
            self.model_name = model_name

        def gerar_embeddings(self, frases):
            return np.ones((len(frases), 4))

        def gerar_embedding(self, frase):
            return np.ones(4)

    class FakePuzzleClassifier:
        def __init__(self, dataset_path, embedding_service):
            self.embedding_service = embedding_service

        def classificar(self, pergunta, modo, threshold=None):
            return {"pergunta": pergunta, "modo": modo, "threshold": threshold}

    class FakeKeywordClassifier:
        def classificar(self, pergunta):
            return {"pergunta": pergunta}

    monkeypatch.setattr(benchmark, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(benchmark, "PuzzleClassifier", FakePuzzleClassifier)
    monkeypatch.setattr(benchmark, "KeywordClassifier", FakeKeywordClassifier)

    resultado = benchmark.executar_benchmark(
        root=ROOT,
        saida=tmp_path,
        repeticoes=2,
        aquecimentos=1,
        por_classe=1,
        seed=42,
    )
    diretorio = resultado["diretorio"]
    consistencia = benchmark.validar_consistencia_execucao(diretorio)

    assert resultado["ambiente"]["status"] == "executado"
    assert consistencia["valido"] is True
    assert resultado["ambiente"]["quantidade_perguntas_medidas"] == 6
    assert set(resultado["resumo"]["id_execucao"]) == {resultado["id_execucao"]}
    assert (diretorio / "amostra_perguntas.csv").exists()
    assert (diretorio / "manifesto_execucao.json").exists()
    assert resultado["execucoes"]["configuracao"].nunique() == 3
    assert set(resultado["execucoes"]["quantidade_perguntas"]) == {6}
