"""Geração dos gráficos e matrizes de confusão dos experimentos.

Todos os gráficos usam exclusivamente números produzidos pelas execuções reais
dos classificadores. O backend ``Agg`` é selecionado para que os arquivos PNG
possam ser gerados pelo terminal, sem janela gráfica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (precisa vir depois de matplotlib.use)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import RESULTS_DIR  # noqa: E402
from .evaluation import ResultadoAvaliacao  # noqa: E402

METRICAS_COMPARACAO = (
    ("acuracia", "Acurácia"),
    ("macro_precision", "Macro Precision"),
    ("macro_recall", "Macro Recall"),
    ("macro_f1", "Macro F1"),
)

ARQUIVO_COMPARACAO = "grafico_comparacao_classificadores.png"
ARQUIVO_THRESHOLDS = "grafico_thresholds.png"
ARQUIVO_DISTRIBUICAO = "grafico_distribuicao_similaridades.png"
PREFIXO_MATRIZ = "matriz_confusao_"


def _preparar_destino(diretorio: Path | str) -> Path:
    destino = Path(diretorio)
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def _rotulo_curto(token: str) -> str:
    """Encurta ``TOKEN_X`` para ``X``, deixando os eixos legíveis."""
    return token.removeprefix("TOKEN_")


def grafico_comparacao_metricas(
    resultados: Sequence[ResultadoAvaliacao],
    caminho: Path | str | None = None,
    diretorio: Path | str = RESULTS_DIR,
) -> Path:
    """Gráfico de barras com as métricas agregadas de cada método."""
    if not resultados:
        raise ValueError("É necessário informar ao menos um resultado para o gráfico.")

    destino = Path(caminho) if caminho else _preparar_destino(diretorio) / ARQUIVO_COMPARACAO
    destino.parent.mkdir(parents=True, exist_ok=True)

    nomes = [resultado.metodo for resultado in resultados]
    posicoes = np.arange(len(METRICAS_COMPARACAO))
    largura = 0.8 / len(resultados)

    figura, eixo = plt.subplots(figsize=(10, 5.5))
    for indice, resultado in enumerate(resultados):
        valores = [resultado.metricas[chave] for chave, _ in METRICAS_COMPARACAO]
        deslocamento = posicoes + indice * largura - 0.4 + largura / 2
        barras = eixo.bar(deslocamento, valores, largura, label=nomes[indice])
        eixo.bar_label(barras, fmt="%.3f", fontsize=7, padding=2)

    conjunto = resultados[0].conjunto
    eixo.set_title(f"Comparação dos classificadores (conjunto de {conjunto})")
    eixo.set_xlabel("Métrica")
    eixo.set_ylabel("Valor (0 a 1)")
    eixo.set_xticks(posicoes)
    eixo.set_xticklabels([rotulo for _, rotulo in METRICAS_COMPARACAO])
    eixo.set_ylim(0, 1.1)
    eixo.grid(axis="y", linestyle=":", alpha=0.6)
    eixo.legend(title="Método", loc="upper left", fontsize=8)
    figura.tight_layout()
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino


def grafico_thresholds(
    tabela: pd.DataFrame,
    threshold_selecionado: float | None = None,
    caminho: Path | str | None = None,
    diretorio: Path | str = RESULTS_DIR,
) -> Path:
    """Gráfico de linhas de acurácia e Macro F1 por threshold (validação)."""
    obrigatorias = {"threshold", "acuracia", "macro_f1"}
    faltantes = sorted(obrigatorias - set(tabela.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes na tabela de thresholds: {faltantes}")
    if tabela.empty:
        raise ValueError("A tabela de thresholds está vazia.")

    destino = Path(caminho) if caminho else _preparar_destino(diretorio) / ARQUIVO_THRESHOLDS
    destino.parent.mkdir(parents=True, exist_ok=True)

    ordenada = tabela.sort_values("threshold")
    figura, eixo = plt.subplots(figsize=(9, 5))
    eixo.plot(ordenada["threshold"], ordenada["acuracia"], marker="o", label="Acurácia")
    eixo.plot(ordenada["threshold"], ordenada["macro_f1"], marker="s", label="Macro F1")

    if threshold_selecionado is not None:
        eixo.axvline(
            threshold_selecionado,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"Threshold selecionado ({threshold_selecionado:.2f})",
        )

    eixo.set_title("Desempenho do modo de rejeição por threshold (conjunto de validação)")
    eixo.set_xlabel("Threshold de similaridade de cosseno")
    eixo.set_ylabel("Valor da métrica (0 a 1)")
    eixo.set_ylim(0, 1.05)
    eixo.grid(linestyle=":", alpha=0.6)
    eixo.legend(loc="lower left", fontsize=8)
    figura.tight_layout()
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino


def grafico_distribuicao_similaridades(
    distribuicao: pd.DataFrame,
    threshold_selecionado: float | None = None,
    caminho: Path | str | None = None,
    diretorio: Path | str = RESULTS_DIR,
) -> Path:
    """Histograma das similaridades máximas frente às cinco intenções do puzzle."""
    obrigatorias = {"similaridade_puzzle", "relacionada"}
    faltantes = sorted(obrigatorias - set(distribuicao.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes na distribuição: {faltantes}")
    if distribuicao.empty:
        raise ValueError("A tabela de distribuição está vazia.")

    destino = Path(caminho) if caminho else _preparar_destino(diretorio) / ARQUIVO_DISTRIBUICAO
    destino.parent.mkdir(parents=True, exist_ok=True)

    relacionadas = distribuicao.loc[distribuicao["relacionada"], "similaridade_puzzle"]
    nao_relacionadas = distribuicao.loc[~distribuicao["relacionada"], "similaridade_puzzle"]
    limites = np.linspace(
        float(distribuicao["similaridade_puzzle"].min()),
        float(distribuicao["similaridade_puzzle"].max()),
        16,
    )

    figura, eixo = plt.subplots(figsize=(9, 5))
    eixo.hist(
        relacionadas,
        bins=limites,
        alpha=0.65,
        label=f"Relacionadas ao puzzle (n={len(relacionadas)})",
        color="tab:blue",
        edgecolor="white",
    )
    eixo.hist(
        nao_relacionadas,
        bins=limites,
        alpha=0.65,
        label=f"Não relacionadas (n={len(nao_relacionadas)})",
        color="tab:orange",
        edgecolor="white",
    )
    if threshold_selecionado is not None:
        eixo.axvline(
            threshold_selecionado,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"Threshold selecionado ({threshold_selecionado:.2f})",
        )

    eixo.set_title(
        "Distribuição da maior similaridade frente às cinco intenções do puzzle"
    )
    eixo.set_xlabel("Similaridade de cosseno máxima")
    eixo.set_ylabel("Quantidade de perguntas")
    eixo.grid(axis="y", linestyle=":", alpha=0.6)
    eixo.legend(fontsize=8)
    figura.tight_layout()
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino


def grafico_matriz_confusao(
    resultado: ResultadoAvaliacao,
    sufixo: str,
    diretorio: Path | str = RESULTS_DIR,
) -> Path:
    """Matriz de confusão anotada de um método, salva em PNG."""
    destino = _preparar_destino(diretorio) / f"{PREFIXO_MATRIZ}{sufixo}.png"
    matriz = np.asarray(resultado.matriz_confusao)
    rotulos = [_rotulo_curto(token) for token in resultado.labels]

    figura, eixo = plt.subplots(figsize=(7.5, 6.5))
    imagem = eixo.imshow(matriz, cmap="Blues", vmin=0)
    figura.colorbar(imagem, ax=eixo, label="Quantidade de perguntas")

    eixo.set_title(f"Matriz de confusão\n{resultado.metodo} | conjunto de {resultado.conjunto}")
    eixo.set_xlabel("Token previsto")
    eixo.set_ylabel("Token esperado")
    eixo.set_xticks(np.arange(len(rotulos)))
    eixo.set_yticks(np.arange(len(rotulos)))
    eixo.set_xticklabels(rotulos, rotation=40, ha="right", fontsize=8)
    eixo.set_yticklabels(rotulos, fontsize=8)

    limite = matriz.max() / 2 if matriz.max() else 0.5
    for linha in range(matriz.shape[0]):
        for coluna in range(matriz.shape[1]):
            valor = int(matriz[linha, coluna])
            eixo.text(
                coluna,
                linha,
                str(valor),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if valor > limite else "black",
            )

    figura.tight_layout()
    figura.savefig(destino, dpi=150)
    plt.close(figura)
    return destino


def gerar_todos(
    resultados: Mapping[str, ResultadoAvaliacao],
    tabela_thresholds: pd.DataFrame | None = None,
    distribuicao: pd.DataFrame | None = None,
    threshold_selecionado: float | None = None,
    diretorio: Path | str = RESULTS_DIR,
) -> Dict[str, Path]:
    """Gera todos os gráficos previstos e devolve os caminhos criados.

    Args:
        resultados: mapa ``sufixo -> ResultadoAvaliacao`` usado nas matrizes de
            confusão e no gráfico comparativo.
    """
    caminhos: Dict[str, Path] = {}
    if resultados:
        caminhos["comparacao"] = grafico_comparacao_metricas(
            list(resultados.values()), diretorio=diretorio
        )
        for sufixo, resultado in resultados.items():
            caminhos[f"matriz_{sufixo}"] = grafico_matriz_confusao(
                resultado, sufixo, diretorio=diretorio
            )
    if tabela_thresholds is not None and not tabela_thresholds.empty:
        caminhos["thresholds"] = grafico_thresholds(
            tabela_thresholds, threshold_selecionado, diretorio=diretorio
        )
    if distribuicao is not None and not distribuicao.empty:
        caminhos["distribuicao"] = grafico_distribuicao_similaridades(
            distribuicao, threshold_selecionado, diretorio=diretorio
        )
    return caminhos


__all__ = [
    "gerar_todos",
    "grafico_comparacao_metricas",
    "grafico_distribuicao_similaridades",
    "grafico_matriz_confusao",
    "grafico_thresholds",
]
