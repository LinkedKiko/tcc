"""Testes do módulo de avaliação, da divisão dos dados e dos relatórios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.classifier import PuzzleClassifier
from src.config import (
    ALL_TOKENS,
    MODO_PADRAO,
    MODO_REJEICAO,
    PUZZLE_TOKENS,
    RANDOM_STATE,
    TOKEN_NAO_RELACIONADO,
)
from src.dataset import (
    carregar_avaliacao,
    dividir_validacao_teste,
    preparar_particoes,
    salvar_particoes,
    verificar_integridade,
)
from src.evaluation import (
    analisar_divergencias,
    analisar_thresholds,
    avaliar_classificador,
    avaliar_embeddings,
    avaliar_predicoes,
    comparar_por_pergunta,
    distribuicao_similaridades,
    salvar_resultado,
    selecionar_threshold,
    tabela_comparativa,
)
from src.keyword_classifier import KeywordClassifier


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_acuracia_e_contagens_corretas() -> None:
    resultado = avaliar_predicoes(
        frases=["a", "b", "c", "d"],
        esperados=["TOKEN_MURAL", "TOKEN_MURAL", "TOKEN_COFRE", "TOKEN_COFRE"],
        previstos=["TOKEN_MURAL", "TOKEN_COFRE", "TOKEN_COFRE", "TOKEN_COFRE"],
    )

    assert resultado.metricas["total"] == 4
    assert resultado.metricas["acertos"] == 3
    assert resultado.metricas["erros"] == 1
    assert resultado.metricas["acuracia"] == pytest.approx(0.75)


def test_metricas_por_classe_conferem_com_calculo_manual() -> None:
    resultado = avaliar_predicoes(
        frases=["a", "b", "c"],
        esperados=["TOKEN_MURAL", "TOKEN_MURAL", "TOKEN_COFRE"],
        previstos=["TOKEN_MURAL", "TOKEN_COFRE", "TOKEN_COFRE"],
        labels=("TOKEN_MURAL", "TOKEN_COFRE"),
    )

    por_classe = resultado.por_classe.set_index("token")
    # TOKEN_MURAL: 1 verdadeiro positivo, 0 falso positivo, 1 falso negativo.
    assert por_classe.loc["TOKEN_MURAL", "precisao"] == pytest.approx(1.0)
    assert por_classe.loc["TOKEN_MURAL", "recall"] == pytest.approx(0.5)
    # TOKEN_COFRE: 1 verdadeiro positivo, 1 falso positivo, 0 falso negativo.
    assert por_classe.loc["TOKEN_COFRE", "precisao"] == pytest.approx(0.5)
    assert por_classe.loc["TOKEN_COFRE", "recall"] == pytest.approx(1.0)
    assert resultado.metricas["macro_f1"] == pytest.approx(
        (2 / 3 + 2 / 3) / 2
    )


def test_classe_ausente_nas_predicoes_nao_gera_divisao_por_zero() -> None:
    resultado = avaliar_predicoes(
        frases=["a", "b"],
        esperados=["TOKEN_MURAL", "TOKEN_MURAL"],
        previstos=["TOKEN_MURAL", "TOKEN_MURAL"],
    )

    por_classe = resultado.por_classe.set_index("token")
    assert por_classe.loc[TOKEN_NAO_RELACIONADO, "precisao"] == 0.0
    assert por_classe.loc[TOKEN_NAO_RELACIONADO, "f1"] == 0.0
    assert np.isfinite(resultado.metricas["macro_f1"])


def test_matriz_confusao_tem_ordem_fixa_dos_tokens() -> None:
    resultado = avaliar_predicoes(
        frases=["a"], esperados=["TOKEN_SIMBOLOS"], previstos=["TOKEN_SIMBOLOS"]
    )

    assert resultado.labels == ALL_TOKENS
    assert resultado.matriz_confusao.shape == (6, 6)
    assert resultado.matriz_confusao[0, 0] == 1


def test_erros_sao_identificados_com_similaridade_e_dificuldade() -> None:
    resultado = avaliar_predicoes(
        frases=["a", "b"],
        esperados=["TOKEN_MURAL", "TOKEN_COFRE"],
        previstos=["TOKEN_MURAL", "TOKEN_LIVRO"],
        similaridades=[0.9, 0.4],
        dificuldades=["facil", "dificil"],
    )

    erros = resultado.erros
    assert len(erros) == 1
    assert erros.loc[0, "frase"] == "b"
    assert erros.loc[0, "token_esperado"] == "TOKEN_COFRE"
    assert erros.loc[0, "token_previsto"] == "TOKEN_LIVRO"
    assert erros.loc[0, "similaridade"] == pytest.approx(0.4)
    assert erros.loc[0, "dificuldade"] == "dificil"


def test_similaridade_ausente_permanece_vazia() -> None:
    resultado = avaliar_predicoes(
        frases=["a"], esperados=["TOKEN_MURAL"], previstos=["TOKEN_MURAL"]
    )

    assert resultado.registros["similaridade"].isna().all()


def test_rotulos_esperados_sao_preservados(dataset_avaliacao_minimo: pd.DataFrame) -> None:
    resultado = avaliar_classificador(
        KeywordClassifier(), dataset_avaliacao_minimo, "baseline", "teste"
    )

    assert resultado.registros["token_esperado"].tolist() == (
        dataset_avaliacao_minimo["token_esperado"].tolist()
    )
    assert resultado.registros["frase"].tolist() == dataset_avaliacao_minimo["frase"].tolist()


def test_quantidade_de_classificacoes_bate_com_o_dataset(
    dataset_avaliacao_minimo: pd.DataFrame,
) -> None:
    resultado = avaliar_classificador(
        KeywordClassifier(), dataset_avaliacao_minimo, "baseline", "teste"
    )

    assert resultado.metricas["total"] == len(dataset_avaliacao_minimo)
    assert len(resultado.registros) == len(dataset_avaliacao_minimo)
    assert resultado.metricas["acertos"] + resultado.metricas["erros"] == len(
        dataset_avaliacao_minimo
    )


def test_dataset_sem_colunas_obrigatorias_gera_erro() -> None:
    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        avaliar_classificador(
            KeywordClassifier(), pd.DataFrame({"texto": ["a"]}), "baseline", "teste"
        )


def test_dataset_vazio_gera_erro() -> None:
    with pytest.raises(ValueError, match="não possui registros"):
        avaliar_classificador(
            KeywordClassifier(),
            pd.DataFrame(columns=["frase", "token_esperado"]),
            "baseline",
            "teste",
        )


def test_tamanhos_incompativeis_geram_erro() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        avaliar_predicoes(frases=["a", "b"], esperados=["X"], previstos=["X"])


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def test_salvar_resultado_gera_os_csvs(tmp_path: Path) -> None:
    resultado = avaliar_predicoes(
        frases=["a", "b"],
        esperados=["TOKEN_MURAL", "TOKEN_COFRE"],
        previstos=["TOKEN_MURAL", "TOKEN_LIVRO"],
        similaridades=[0.9, None],
        dificuldades=["facil", "media"],
    )

    caminhos = salvar_resultado(resultado, "teste_unitario", tmp_path)

    for caminho in caminhos.values():
        assert caminho.exists()

    detalhado = pd.read_csv(caminhos["detalhado"])
    assert list(detalhado.columns) == [
        "frase",
        "token_esperado",
        "token_previsto",
        "similaridade",
        "correto",
        "dificuldade",
    ]
    assert pd.isna(detalhado.loc[1, "similaridade"])
    assert len(pd.read_csv(caminhos["erros"])) == 1


# ---------------------------------------------------------------------------
# Divisão determinística dos dados
# ---------------------------------------------------------------------------
def test_dataset_de_avaliacao_tem_cem_frases(caminho_avaliacao: Path) -> None:
    dataset = carregar_avaliacao(caminho_avaliacao)

    assert len(dataset) == 100
    assert list(dataset.columns) == ["frase", "token_esperado", "dificuldade"]


def test_distribuicao_esperada_do_dataset(caminho_avaliacao: Path) -> None:
    dataset = carregar_avaliacao(caminho_avaliacao)
    contagem = dataset["token_esperado"].value_counts()

    for token in PUZZLE_TOKENS:
        assert contagem[token] == 15
    assert contagem[TOKEN_NAO_RELACIONADO] == 25


def test_avaliacao_nao_repete_frases_da_referencia(caminho_avaliacao: Path) -> None:
    particoes = preparar_particoes(caminho_avaliacao)

    integridade = verificar_integridade(particoes)

    assert integridade["frases_em_comum"] == []


def test_divisao_respeita_as_proporcoes(caminho_avaliacao: Path) -> None:
    particoes = preparar_particoes(caminho_avaliacao)

    assert len(particoes.validacao) == 40
    assert len(particoes.teste) == 60
    assert len(particoes.completo) == 100


def test_divisao_e_estratificada(caminho_avaliacao: Path) -> None:
    particoes = preparar_particoes(caminho_avaliacao)

    for token in PUZZLE_TOKENS:
        assert (particoes.validacao["token_esperado"] == token).sum() == 6
        assert (particoes.teste["token_esperado"] == token).sum() == 9
    assert (particoes.validacao["token_esperado"] == TOKEN_NAO_RELACIONADO).sum() == 10
    assert (particoes.teste["token_esperado"] == TOKEN_NAO_RELACIONADO).sum() == 15


def test_divisao_e_deterministica(caminho_avaliacao: Path) -> None:
    primeira = preparar_particoes(caminho_avaliacao)
    segunda = preparar_particoes(caminho_avaliacao)

    assert primeira.validacao["frase"].tolist() == segunda.validacao["frase"].tolist()
    assert primeira.teste["frase"].tolist() == segunda.teste["frase"].tolist()
    assert primeira.random_state == RANDOM_STATE


def test_sementes_diferentes_produzem_particoes_diferentes(caminho_avaliacao: Path) -> None:
    primeira = preparar_particoes(caminho_avaliacao, random_state=1)
    segunda = preparar_particoes(caminho_avaliacao, random_state=2)

    assert primeira.validacao["frase"].tolist() != segunda.validacao["frase"].tolist()


def test_nao_existe_sobreposicao_entre_validacao_e_teste(caminho_avaliacao: Path) -> None:
    particoes = preparar_particoes(caminho_avaliacao)

    validacao = set(particoes.validacao["frase"])
    teste = set(particoes.teste["frase"])
    assert validacao & teste == set()
    assert validacao | teste == set(particoes.completo["frase"])


def test_resumo_das_particoes_soma_o_total(caminho_avaliacao: Path) -> None:
    resumo = preparar_particoes(caminho_avaliacao).resumo()

    total = resumo.loc[resumo["token_esperado"] == "TOTAL"].iloc[0]
    assert total["completo"] == 100
    assert total["validacao"] + total["teste"] == 100


def test_salvar_particoes_registra_os_arquivos(
    caminho_avaliacao: Path, tmp_path: Path
) -> None:
    particoes = preparar_particoes(caminho_avaliacao)

    caminhos = salvar_particoes(particoes, tmp_path)

    for caminho in caminhos.values():
        assert caminho.exists()
    validacao = pd.read_csv(caminhos["validacao"])
    assert validacao["particao"].unique().tolist() == ["validacao"]
    assert len(validacao) == 40


def test_divisao_rejeita_classe_com_uma_unica_frase() -> None:
    dataset = pd.DataFrame(
        {
            "frase": ["a", "b", "c"],
            "token_esperado": ["TOKEN_MURAL", "TOKEN_MURAL", "TOKEN_COFRE"],
            "dificuldade": ["facil"] * 3,
        }
    )

    with pytest.raises(ValueError, match="ao menos duas frases"):
        dividir_validacao_teste(dataset)


def test_dataset_com_frases_repetidas_gera_erro(tmp_path: Path) -> None:
    caminho = tmp_path / "repetido.csv"
    caminho.write_text(
        "frase,token_esperado,dificuldade\nigual,TOKEN_MURAL,facil\nigual,TOKEN_COFRE,facil\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frases repetidas"):
        carregar_avaliacao(caminho)


def test_dataset_com_token_desconhecido_gera_erro(tmp_path: Path) -> None:
    caminho = tmp_path / "token.csv"
    caminho.write_text(
        "frase,token_esperado\nalgo,TOKEN_INVENTADO\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Tokens esperados desconhecidos"):
        carregar_avaliacao(caminho)


def test_coluna_dificuldade_opcional_recebe_valor_padrao(tmp_path: Path) -> None:
    caminho = tmp_path / "sem_dificuldade.csv"
    caminho.write_text(
        "frase,token_esperado\nalgo sobre o mural,TOKEN_MURAL\n", encoding="utf-8"
    )

    dataset = carregar_avaliacao(caminho)

    assert dataset.loc[0, "dificuldade"] == "nao_informada"


# ---------------------------------------------------------------------------
# Comparação entre métodos
# ---------------------------------------------------------------------------
def _resultado_fixo(previstos, metodo, similaridades=None):
    return avaliar_predicoes(
        frases=["a", "b", "c", "d"],
        esperados=["TOKEN_MURAL", "TOKEN_COFRE", "TOKEN_LIVRO", TOKEN_NAO_RELACIONADO],
        previstos=previstos,
        similaridades=similaridades,
        metodo=metodo,
    )


def test_comparacao_por_pergunta_tem_as_colunas_exigidas() -> None:
    keywords = _resultado_fixo(
        ["TOKEN_MURAL", "TOKEN_LIVRO", "TOKEN_LIVRO", TOKEN_NAO_RELACIONADO], "keywords"
    )
    embeddings = _resultado_fixo(
        ["TOKEN_COFRE", "TOKEN_COFRE", "TOKEN_LIVRO", "TOKEN_MURAL"],
        "embeddings",
        similaridades=[0.5, 0.6, 0.7, 0.8],
    )

    comparacao = comparar_por_pergunta(keywords, embeddings)

    assert list(comparacao.columns)[:7] == [
        "frase",
        "token_esperado",
        "token_keywords",
        "token_embeddings",
        "similaridade",
        "acerto_keywords",
        "acerto_embeddings",
    ]


def test_divergencias_separam_os_quatro_casos() -> None:
    keywords = _resultado_fixo(
        ["TOKEN_MURAL", "TOKEN_LIVRO", "TOKEN_LIVRO", TOKEN_NAO_RELACIONADO], "keywords"
    )
    embeddings = _resultado_fixo(
        ["TOKEN_COFRE", "TOKEN_COFRE", "TOKEN_LIVRO", "TOKEN_MURAL"],
        "embeddings",
        similaridades=[0.5, 0.6, 0.7, 0.8],
    )

    casos = analisar_divergencias(comparar_por_pergunta(keywords, embeddings))

    assert casos["caso_a"]["frase"].tolist() == ["b"]
    assert casos["caso_b"]["frase"].tolist() == ["a", "d"]
    assert casos["caso_c"]["frase"].tolist() == ["c"]
    assert casos["caso_d"].empty


def test_comparacao_exige_as_mesmas_perguntas() -> None:
    keywords = _resultado_fixo(["TOKEN_MURAL"] * 4, "keywords")
    outro = avaliar_predicoes(
        frases=["x", "y", "z", "w"],
        esperados=["TOKEN_MURAL"] * 4,
        previstos=["TOKEN_MURAL"] * 4,
        metodo="embeddings",
    )

    with pytest.raises(ValueError, match="perguntas diferentes"):
        comparar_por_pergunta(keywords, outro)


def test_tabela_comparativa_lista_todos_os_metodos() -> None:
    primeiro = _resultado_fixo(["TOKEN_MURAL"] * 4, "A")
    segundo = _resultado_fixo(["TOKEN_COFRE"] * 4, "B")

    tabela = tabela_comparativa([primeiro, segundo])

    assert tabela["metodo"].tolist() == ["A", "B"]
    assert {"acuracia", "macro_f1", "macro_precision", "macro_recall"} <= set(tabela.columns)


# ---------------------------------------------------------------------------
# Análise de thresholds (com embeddings determinísticos)
# ---------------------------------------------------------------------------
def test_tabela_de_thresholds_tem_as_colunas_exigidas(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    tabela = analisar_thresholds(
        classificador, dataset_avaliacao_minimo, thresholds=[0.3, 0.6, 0.9]
    )

    assert list(tabela.columns)[:7] == [
        "threshold",
        "acuracia",
        "macro_f1",
        "aceitas_corretamente",
        "rejeitadas_corretamente",
        "falsas_aceitacoes",
        "falsas_rejeicoes",
    ]
    assert tabela["threshold"].tolist() == [0.3, 0.6, 0.9]


def test_contagens_de_rejeicao_somam_o_total(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    tabela = analisar_thresholds(
        classificador, dataset_avaliacao_minimo, thresholds=[0.5]
    )

    linha = tabela.iloc[0]
    soma = (
        linha["aceitas_corretamente"]
        + linha["falsas_rejeicoes"]
        + linha["rejeitadas_corretamente"]
        + linha["falsas_aceitacoes"]
    )
    assert soma == len(dataset_avaliacao_minimo)


def test_classificacao_correta_nao_excede_as_aceitas(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    tabela = analisar_thresholds(
        classificador, dataset_avaliacao_minimo, thresholds=[0.2, 0.5, 0.8]
    )

    assert (tabela["classificadas_corretamente"] <= tabela["aceitas_corretamente"]).all()


def test_threshold_acima_das_similaridades_rejeita_tudo(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    # Matriz controlada: todas as similaridades valem 0.20, abaixo do limiar.
    matriz = np.full(
        (len(dataset_avaliacao_minimo), len(classificador.frases_referencia)), 0.20
    )

    tabela = analisar_thresholds(
        classificador, dataset_avaliacao_minimo, thresholds=[0.50], matriz=matriz
    )

    linha = tabela.iloc[0]
    assert linha["aceitas_corretamente"] == 0
    assert linha["falsas_aceitacoes"] == 0
    assert linha["taxa_rejeicao_correta"] == pytest.approx(1.0)
    assert linha["taxa_rejeicao_incorreta"] == pytest.approx(1.0)


def test_threshold_abaixo_das_similaridades_aceita_tudo(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    matriz = np.full(
        (len(dataset_avaliacao_minimo), len(classificador.frases_referencia)), 0.80
    )

    tabela = analisar_thresholds(
        classificador, dataset_avaliacao_minimo, thresholds=[0.50], matriz=matriz
    )

    linha = tabela.iloc[0]
    assert linha["falsas_rejeicoes"] == 0
    assert linha["rejeitadas_corretamente"] == 0
    assert linha["aceitas_corretamente"] == 5
    assert linha["falsas_aceitacoes"] == 1


def test_aceitacoes_nao_aumentam_com_threshold_maior(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    tabela = analisar_thresholds(
        classificador, dataset_avaliacao_minimo, thresholds=[0.2, 0.5, 0.8, 1.0]
    )

    aceitas = (tabela["aceitas_corretamente"] + tabela["falsas_aceitacoes"]).tolist()
    assert aceitas == sorted(aceitas, reverse=True)


def test_selecao_do_threshold_usa_maior_macro_f1() -> None:
    tabela = pd.DataFrame(
        {
            "threshold": [0.4, 0.5, 0.6],
            "acuracia": [0.5, 0.9, 0.7],
            "macro_f1": [0.4, 0.8, 0.6],
        }
    )

    selecao = selecionar_threshold(tabela)

    assert selecao["threshold"] == pytest.approx(0.5)
    assert selecao["macro_f1"] == pytest.approx(0.8)
    assert selecao["empatados"] == 1


def test_empate_de_macro_f1_e_resolvido_pela_acuracia() -> None:
    tabela = pd.DataFrame(
        {
            "threshold": [0.4, 0.5, 0.6],
            "acuracia": [0.6, 0.8, 0.7],
            "macro_f1": [0.8, 0.8, 0.5],
        }
    )

    assert selecionar_threshold(tabela)["threshold"] == pytest.approx(0.5)


def test_empate_total_e_resolvido_pelo_menor_threshold() -> None:
    tabela = pd.DataFrame(
        {
            "threshold": [0.45, 0.55, 0.65],
            "acuracia": [0.8, 0.8, 0.5],
            "macro_f1": [0.7, 0.7, 0.4],
        }
    )

    selecao = selecionar_threshold(tabela)

    assert selecao["threshold"] == pytest.approx(0.45)
    assert selecao["empatados"] == 2


def test_selecao_rejeita_tabela_vazia() -> None:
    with pytest.raises(ValueError, match="vazia"):
        selecionar_threshold(
            pd.DataFrame(columns=["threshold", "acuracia", "macro_f1"])
        )


def test_threshold_selecionado_apenas_na_validacao(caminho_avaliacao: Path, classificador) -> None:
    """A grade de limiares roda sobre a validação e não toca o conjunto de teste."""
    particoes = preparar_particoes(caminho_avaliacao)

    tabela = analisar_thresholds(
        classificador, particoes.validacao, thresholds=[0.4, 0.6, 0.8]
    )

    total_validacao = len(particoes.validacao)
    colunas = [
        "aceitas_corretamente",
        "falsas_rejeicoes",
        "rejeitadas_corretamente",
        "falsas_aceitacoes",
    ]
    assert (tabela[colunas].sum(axis=1) == total_validacao).all()
    assert total_validacao == 40
    assert len(particoes.teste) == 60


# ---------------------------------------------------------------------------
# Avaliação dos modos do classificador de embeddings
# ---------------------------------------------------------------------------
def test_avaliar_embeddings_sem_threshold_registra_similaridade(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    resultado = avaliar_embeddings(
        classificador, dataset_avaliacao_minimo, conjunto="teste", modo=MODO_PADRAO
    )

    assert resultado.registros["similaridade"].notna().all()
    assert resultado.detalhes["threshold"] is None
    assert "similaridade_puzzle" in resultado.registros.columns


def test_modo_rejeicao_abaixo_do_limiar_preve_tudo_como_nao_relacionado(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    matriz = np.full(
        (len(dataset_avaliacao_minimo), len(classificador.frases_referencia)), 0.10
    )

    resultado = avaliar_embeddings(
        classificador,
        dataset_avaliacao_minimo,
        conjunto="teste",
        modo=MODO_REJEICAO,
        threshold=0.60,
        matriz=matriz,
    )

    assert set(resultado.registros["token_previsto"]) == {TOKEN_NAO_RELACIONADO}
    assert resultado.metricas["acertos"] == 1  # apenas a pergunta nao relacionada


def test_distribuicao_de_similaridades_marca_perguntas_relacionadas(
    classificador: PuzzleClassifier, dataset_avaliacao_minimo: pd.DataFrame
) -> None:
    distribuicao = distribuicao_similaridades(classificador, dataset_avaliacao_minimo)

    assert distribuicao["relacionada"].sum() == 5
    assert (~distribuicao["relacionada"]).sum() == 1
    assert distribuicao["similaridade_puzzle"].between(-1.0, 1.0).all()
