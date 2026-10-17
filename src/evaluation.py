"""Avaliação automatizada dos classificadores.

O módulo é independente de qual classificador está sendo avaliado: basta que o
objeto exponha ``classificar(pergunta)`` devolvendo um dicionário com as chaves
``token`` e ``similaridade``. Quando o método não produz similaridade de
cosseno, como o baseline por palavras-chave, o campo permanece vazio.

Separação de responsabilidades
------------------------------
* o cálculo das métricas vive em :func:`avaliar_predicoes`;
* a apresentação no terminal vive em :func:`imprimir_relatorio`;
* a persistência vive em :func:`salvar_resultado`.

Dessa forma o mesmo cálculo alimenta o terminal, os CSVs, os gráficos e o
relatório em Markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .classifier import PuzzleClassifier
from .config import (
    ALL_TOKENS,
    MODO_PADRAO,
    MODO_REJEICAO,
    PUZZLE_TOKENS,
    RESULTS_DIR,
    THRESHOLDS,
)

COLUNAS_DETALHE = (
    "frase",
    "token_esperado",
    "token_previsto",
    "similaridade",
    "correto",
    "dificuldade",
)

COLUNAS_COMPARACAO = (
    "frase",
    "token_esperado",
    "token_keywords",
    "token_embeddings",
    "similaridade",
    "acerto_keywords",
    "acerto_embeddings",
)


class Classificador(Protocol):
    """Interface mínima aceita pelo módulo de avaliação."""

    def classificar(self, pergunta: str) -> Mapping[str, Any]:  # pragma: no cover
        ...


@dataclass
class ResultadoAvaliacao:
    """Agrupa as predições, as métricas e a matriz de confusão de um método."""

    metodo: str
    conjunto: str
    registros: pd.DataFrame
    metricas: Dict[str, float]
    por_classe: pd.DataFrame
    matriz_confusao: np.ndarray
    labels: Tuple[str, ...]
    detalhes: Dict[str, Any] = field(default_factory=dict)

    @property
    def erros(self) -> pd.DataFrame:
        """Perguntas classificadas incorretamente, com similaridade e dificuldade."""
        return self.registros.loc[~self.registros["correto"]].reset_index(drop=True)

    @property
    def acertos(self) -> pd.DataFrame:
        """Perguntas classificadas corretamente."""
        return self.registros.loc[self.registros["correto"]].reset_index(drop=True)

    def matriz_confusao_df(self) -> pd.DataFrame:
        """Matriz de confusão como DataFrame rotulado (linhas = esperado)."""
        return pd.DataFrame(
            self.matriz_confusao,
            index=[f"real_{token}" for token in self.labels],
            columns=[f"previsto_{token}" for token in self.labels],
        )


# ---------------------------------------------------------------------------
# Cálculo das métricas
# ---------------------------------------------------------------------------
def avaliar_predicoes(
    frases: Sequence[str],
    esperados: Sequence[str],
    previstos: Sequence[str],
    similaridades: Optional[Sequence[Optional[float]]] = None,
    dificuldades: Optional[Sequence[str]] = None,
    metodo: str = "classificador",
    conjunto: str = "teste",
    labels: Sequence[str] = ALL_TOKENS,
    detalhes: Optional[Dict[str, Any]] = None,
) -> ResultadoAvaliacao:
    """Calcula as métricas de desempenho a partir de rótulos e predições.

    As divisões por zero das métricas por classe recebem valor 0, o que ocorre
    quando alguma classe não aparece nas predições de uma partição.
    """
    total = len(frases)
    if total == 0:
        raise ValueError("Não há perguntas para avaliar.")
    if not (len(esperados) == len(previstos) == total):
        raise ValueError("Frases, rótulos esperados e predições precisam ter o mesmo tamanho.")

    if similaridades is None:
        similaridades = [None] * total
    elif len(similaridades) != total:
        raise ValueError("A lista de similaridades não corresponde à quantidade de frases.")
    if dificuldades is None:
        dificuldades = ["nao_informada"] * total
    elif len(dificuldades) != total:
        raise ValueError("A lista de dificuldades não corresponde à quantidade de frases.")

    labels = tuple(labels)
    registros = pd.DataFrame(
        {
            "frase": list(frases),
            "token_esperado": list(esperados),
            "token_previsto": list(previstos),
            "similaridade": [
                None if valor is None else float(valor) for valor in similaridades
            ],
            "dificuldade": list(dificuldades),
        }
    )
    registros["correto"] = registros["token_esperado"] == registros["token_previsto"]
    registros = registros[list(COLUNAS_DETALHE)]

    acuracia = float(accuracy_score(registros["token_esperado"], registros["token_previsto"]))
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        registros["token_esperado"],
        registros["token_previsto"],
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        registros["token_esperado"],
        registros["token_previsto"],
        labels=labels,
        average="macro",
        zero_division=0,
    )

    por_classe = pd.DataFrame(
        {
            "token": labels,
            "precisao": precisao,
            "recall": recall,
            "f1": f1,
            "suporte": suporte.astype(int),
        }
    )

    acertos = int(registros["correto"].sum())
    metricas = {
        "total": int(total),
        "acertos": acertos,
        "erros": int(total - acertos),
        "acuracia": acuracia,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }

    matriz = confusion_matrix(
        registros["token_esperado"], registros["token_previsto"], labels=labels
    )

    return ResultadoAvaliacao(
        metodo=metodo,
        conjunto=conjunto,
        registros=registros,
        metricas=metricas,
        por_classe=por_classe,
        matriz_confusao=matriz,
        labels=labels,
        detalhes=dict(detalhes or {}),
    )


def avaliar_classificador(
    classificador: Classificador,
    dataset: pd.DataFrame,
    metodo: str,
    conjunto: str,
    labels: Sequence[str] = ALL_TOKENS,
) -> ResultadoAvaliacao:
    """Avalia qualquer classificador sobre um DataFrame rotulado.

    Usa ``classificar_em_lote`` quando disponível, para que o classificador de
    embeddings gere os vetores das perguntas em uma única chamada ao modelo.
    """
    _validar_dataset(dataset)
    frases = dataset["frase"].astype(str).tolist()

    em_lote = getattr(classificador, "classificar_em_lote", None)
    if callable(em_lote):
        resultados: List[Mapping[str, Any]] = list(em_lote(frases))
    else:
        resultados = [classificador.classificar(frase) for frase in frases]

    previstos = [str(resultado["token"]) for resultado in resultados]
    similaridades = [resultado.get("similaridade") for resultado in resultados]

    return avaliar_predicoes(
        frases=frases,
        esperados=dataset["token_esperado"].astype(str).tolist(),
        previstos=previstos,
        similaridades=similaridades,
        dificuldades=dataset.get("dificuldade", pd.Series(["nao_informada"] * len(dataset))).tolist(),
        metodo=metodo,
        conjunto=conjunto,
        labels=labels,
    )


def avaliar_embeddings(
    classificador: PuzzleClassifier,
    dataset: pd.DataFrame,
    conjunto: str,
    modo: str = MODO_PADRAO,
    threshold: Optional[float] = None,
    matriz: Optional[np.ndarray] = None,
    metodo: Optional[str] = None,
    labels: Sequence[str] = ALL_TOKENS,
) -> ResultadoAvaliacao:
    """Avalia o classificador de embeddings reaproveitando a matriz de similaridades."""
    _validar_dataset(dataset)
    frases = dataset["frase"].astype(str).tolist()
    if matriz is None:
        matriz = classificador.matriz_similaridade(frases)

    resultados = classificador.classificar_de_matriz(frases, matriz, modo, threshold)
    if metodo is None:
        metodo = (
            "Embeddings sem threshold"
            if modo == MODO_PADRAO and threshold is None
            else f"Embeddings {modo} (threshold {threshold})"
        )

    avaliacao = avaliar_predicoes(
        frases=frases,
        esperados=dataset["token_esperado"].astype(str).tolist(),
        previstos=[str(resultado["token"]) for resultado in resultados],
        similaridades=[resultado["similaridade"] for resultado in resultados],
        dificuldades=dataset.get(
            "dificuldade", pd.Series(["nao_informada"] * len(dataset))
        ).tolist(),
        metodo=metodo,
        conjunto=conjunto,
        labels=labels,
        detalhes={"modo": modo, "threshold": threshold},
    )
    avaliacao.registros["similaridade_puzzle"] = classificador.similaridade_maxima_puzzle(matriz)
    return avaliacao


# ---------------------------------------------------------------------------
# Apresentação
# ---------------------------------------------------------------------------
def imprimir_relatorio(resultado: ResultadoAvaliacao, mostrar_erros: bool = True) -> None:
    """Imprime no terminal o relatório de uma avaliação."""
    metricas = resultado.metricas
    print("\n========================================")
    print("AVALIACAO DO CLASSIFICADOR")
    print("========================================")
    print(f"\nMetodo: {resultado.metodo}")
    print(f"Conjunto: {resultado.conjunto}")
    if resultado.detalhes.get("threshold") is not None:
        print(f"Threshold aplicado: {resultado.detalhes['threshold']:.2f}")
    print(f"\nTotal de perguntas: {metricas['total']}")
    print(f"Acertos: {metricas['acertos']}")
    print(f"Erros: {metricas['erros']}")
    print(f"\nAcuracia: {metricas['acuracia']:.2%}")
    print(f"Macro Precision: {metricas['macro_precision']:.4f}")
    print(f"Macro Recall: {metricas['macro_recall']:.4f}")
    print(f"Macro F1: {metricas['macro_f1']:.4f}")

    print("\n========================================")
    print("RESULTADOS POR TOKEN")
    print("========================================")
    print(f"\n{'Token':<24}{'Prec.':>8}{'Recall':>8}{'F1':>8}{'Sup.':>6}")
    for linha in resultado.por_classe.itertuples(index=False):
        print(
            f"{linha.token:<24}{linha.precisao:>8.4f}{linha.recall:>8.4f}"
            f"{linha.f1:>8.4f}{linha.suporte:>6d}"
        )

    print("\n========================================")
    print("MATRIZ DE CONFUSAO")
    print("========================================")
    print("(linhas = token esperado, colunas = token previsto)\n")
    print(resultado.matriz_confusao_df().to_string())

    if mostrar_erros:
        print("\n========================================")
        print("ERROS DE CLASSIFICACAO")
        print("========================================")
        erros = resultado.erros
        if erros.empty:
            print("\nNenhum erro encontrado.")
        else:
            for linha in erros.itertuples(index=False):
                similaridade = (
                    "n/d" if pd.isna(linha.similaridade) else f"{linha.similaridade:.4f}"
                )
                print(f"\nPergunta: {linha.frase}")
                print(f"  Token esperado: {linha.token_esperado}")
                print(f"  Token previsto: {linha.token_previsto}")
                print(f"  Similaridade: {similaridade}")
                print(f"  Dificuldade: {linha.dificuldade}")


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def salvar_resultado(
    resultado: ResultadoAvaliacao,
    prefixo: str,
    diretorio: Path | str = RESULTS_DIR,
) -> Dict[str, Path]:
    """Salva predições, erros, métricas e matriz de confusão em ``results/``."""
    destino = Path(diretorio)
    destino.mkdir(parents=True, exist_ok=True)

    caminhos = {
        "detalhado": destino / f"{prefixo}_detalhado.csv",
        "erros": destino / f"{prefixo}_erros.csv",
        "metricas": destino / f"{prefixo}_metricas.csv",
        "matriz_confusao": destino / f"{prefixo}_matriz_confusao.csv",
    }

    resultado.registros.to_csv(caminhos["detalhado"], index=False, encoding="utf-8")
    resultado.erros.to_csv(caminhos["erros"], index=False, encoding="utf-8")

    metricas = pd.DataFrame([resultado.metricas])
    metricas.insert(0, "metodo", resultado.metodo)
    metricas.insert(1, "conjunto", resultado.conjunto)
    metricas.insert(2, "threshold", resultado.detalhes.get("threshold"))
    metricas.to_csv(caminhos["metricas"], index=False, encoding="utf-8")

    resultado.por_classe.assign(
        metodo=resultado.metodo, conjunto=resultado.conjunto
    ).to_csv(destino / f"{prefixo}_por_classe.csv", index=False, encoding="utf-8")
    caminhos["por_classe"] = destino / f"{prefixo}_por_classe.csv"

    resultado.matriz_confusao_df().to_csv(caminhos["matriz_confusao"], encoding="utf-8")
    return caminhos


# ---------------------------------------------------------------------------
# Comparação entre métodos
# ---------------------------------------------------------------------------
def tabela_comparativa(resultados: Sequence[ResultadoAvaliacao]) -> pd.DataFrame:
    """Tabela com as métricas agregadas de cada método avaliado."""
    if not resultados:
        raise ValueError("É necessário informar ao menos um resultado.")
    linhas = []
    for resultado in resultados:
        linha = {"metodo": resultado.metodo, "conjunto": resultado.conjunto}
        linha["threshold"] = resultado.detalhes.get("threshold")
        linha.update(resultado.metricas)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def comparar_por_pergunta(
    resultado_keywords: ResultadoAvaliacao,
    resultado_embeddings: ResultadoAvaliacao,
) -> pd.DataFrame:
    """Alinha as predições dos dois métodos pergunta por pergunta."""
    keywords = resultado_keywords.registros
    embeddings = resultado_embeddings.registros
    if list(keywords["frase"]) != list(embeddings["frase"]):
        raise ValueError("Os métodos foram avaliados sobre perguntas diferentes.")
    if list(keywords["token_esperado"]) != list(embeddings["token_esperado"]):
        raise ValueError("Os rótulos esperados diferem entre as avaliações.")

    comparacao = pd.DataFrame(
        {
            "frase": keywords["frase"].tolist(),
            "token_esperado": keywords["token_esperado"].tolist(),
            "token_keywords": keywords["token_previsto"].tolist(),
            "token_embeddings": embeddings["token_previsto"].tolist(),
            "similaridade": embeddings["similaridade"].tolist(),
            "acerto_keywords": keywords["correto"].tolist(),
            "acerto_embeddings": embeddings["correto"].tolist(),
        }
    )
    comparacao["dificuldade"] = keywords["dificuldade"].tolist()
    return comparacao[list(COLUNAS_COMPARACAO) + ["dificuldade"]]


def analisar_divergencias(comparacao: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Separa a comparação nos quatro casos de concordância e divergência.

    * ``caso_a``: embeddings acertaram e palavras-chave erraram.
    * ``caso_b``: palavras-chave acertaram e embeddings erraram.
    * ``caso_c``: ambos acertaram.
    * ``caso_d``: ambos erraram.
    """
    faltantes = sorted({"acerto_keywords", "acerto_embeddings"} - set(comparacao.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes na comparação: {faltantes}")

    keywords = comparacao["acerto_keywords"].astype(bool)
    embeddings = comparacao["acerto_embeddings"].astype(bool)
    return {
        "caso_a": comparacao.loc[embeddings & ~keywords].reset_index(drop=True),
        "caso_b": comparacao.loc[keywords & ~embeddings].reset_index(drop=True),
        "caso_c": comparacao.loc[keywords & embeddings].reset_index(drop=True),
        "caso_d": comparacao.loc[~keywords & ~embeddings].reset_index(drop=True),
    }


DESCRICAO_CASOS = {
    "caso_a": "Embeddings acertaram e palavras-chave erraram",
    "caso_b": "Palavras-chave acertaram e embeddings erraram",
    "caso_c": "Ambos acertaram",
    "caso_d": "Ambos erraram",
}


def imprimir_divergencias(
    casos: Mapping[str, pd.DataFrame],
    limite_exemplos: int = 3,
) -> None:
    """Imprime a contagem e alguns exemplos reais de cada caso.

    Categorias vazias são declaradas como vazias; nenhum exemplo é fabricado.
    """
    print("\n========================================")
    print("ANALISE DE DIVERGENCIAS")
    print("========================================")
    for chave, descricao in DESCRICAO_CASOS.items():
        bloco = casos.get(chave, pd.DataFrame())
        print(f"\n{chave.upper()} - {descricao}: {len(bloco)} pergunta(s)")
        if bloco.empty:
            print("  Nenhuma pergunta nesta categoria.")
            continue
        for linha in bloco.head(limite_exemplos).itertuples(index=False):
            similaridade = (
                "n/d" if pd.isna(linha.similaridade) else f"{linha.similaridade:.4f}"
            )
            print(f"  - {linha.frase}")
            print(
                f"    esperado={linha.token_esperado} | keywords={linha.token_keywords}"
                f" | embeddings={linha.token_embeddings} | sim={similaridade}"
            )


def dificuldade_por_metodo(resultado: ResultadoAvaliacao) -> pd.DataFrame:
    """Acurácia por nível de dificuldade, útil para analisar onde cada método falha."""
    agrupado = resultado.registros.groupby("dificuldade")["correto"]
    tabela = agrupado.agg(total="size", acertos="sum").reset_index()
    tabela["acuracia"] = tabela["acertos"] / tabela["total"]
    tabela.insert(0, "metodo", resultado.metodo)
    return tabela


# ---------------------------------------------------------------------------
# Análise de thresholds
# ---------------------------------------------------------------------------
COLUNAS_THRESHOLD = (
    "threshold",
    "acuracia",
    "macro_f1",
    "aceitas_corretamente",
    "rejeitadas_corretamente",
    "falsas_aceitacoes",
    "falsas_rejeicoes",
)


def analisar_thresholds(
    classificador: PuzzleClassifier,
    dataset: pd.DataFrame,
    thresholds: Iterable[float] = THRESHOLDS,
    matriz: Optional[np.ndarray] = None,
    labels: Sequence[str] = ALL_TOKENS,
) -> pd.DataFrame:
    """Mede o comportamento do modo de rejeição para cada limiar da grade.

    A matriz de similaridades é calculada uma única vez e reutilizada em todos
    os limiares, de modo que o modelo é chamado apenas uma vez.

    As colunas distinguem a decisão de aceitar ou rejeitar da classificação
    entre as cinco intenções válidas:

    * ``aceitas_corretamente``: perguntas relacionadas que foram aceitas;
    * ``classificadas_corretamente``: perguntas relacionadas aceitas **e** com
      o token correto;
    * ``falsas_rejeicoes``: perguntas relacionadas rejeitadas;
    * ``rejeitadas_corretamente``: perguntas não relacionadas rejeitadas;
    * ``falsas_aceitacoes``: perguntas não relacionadas aceitas.
    """
    _validar_dataset(dataset)
    thresholds = [float(valor) for valor in thresholds]
    if not thresholds:
        raise ValueError("É necessário informar ao menos um threshold.")

    frases = dataset["frase"].astype(str).tolist()
    esperados = dataset["token_esperado"].astype(str).tolist()
    if matriz is None:
        matriz = classificador.matriz_similaridade(frases)

    relacionadas = np.array([token in PUZZLE_TOKENS for token in esperados])
    total_relacionadas = int(relacionadas.sum())
    total_nao_relacionadas = int((~relacionadas).sum())

    linhas = []
    for threshold in sorted(thresholds):
        resultados = classificador.classificar_de_matriz(
            frases, matriz, MODO_REJEICAO, threshold
        )
        previstos = np.array([resultado["token"] for resultado in resultados])
        aceita = np.array([not resultado["rejeitada"] for resultado in resultados])
        corretos = previstos == np.array(esperados)

        avaliacao = avaliar_predicoes(
            frases=frases,
            esperados=esperados,
            previstos=previstos.tolist(),
            similaridades=[r["similaridade"] for r in resultados],
            dificuldades=dataset.get(
                "dificuldade", pd.Series(["nao_informada"] * len(dataset))
            ).tolist(),
            metodo=f"Embeddings com threshold {threshold:.2f}",
            conjunto="validacao",
            labels=labels,
        )

        aceitas_corretamente = int((relacionadas & aceita).sum())
        falsas_rejeicoes = int((relacionadas & ~aceita).sum())
        rejeitadas_corretamente = int((~relacionadas & ~aceita).sum())
        falsas_aceitacoes = int((~relacionadas & aceita).sum())
        classificadas_corretamente = int((relacionadas & aceita & corretos).sum())

        linhas.append(
            {
                "threshold": threshold,
                "acuracia": avaliacao.metricas["acuracia"],
                "macro_f1": avaliacao.metricas["macro_f1"],
                "aceitas_corretamente": aceitas_corretamente,
                "rejeitadas_corretamente": rejeitadas_corretamente,
                "falsas_aceitacoes": falsas_aceitacoes,
                "falsas_rejeicoes": falsas_rejeicoes,
                "classificadas_corretamente": classificadas_corretamente,
                "macro_precision": avaliacao.metricas["macro_precision"],
                "macro_recall": avaliacao.metricas["macro_recall"],
                "total_relacionadas": total_relacionadas,
                "total_nao_relacionadas": total_nao_relacionadas,
                "taxa_rejeicao_correta": (
                    rejeitadas_corretamente / total_nao_relacionadas
                    if total_nao_relacionadas
                    else float("nan")
                ),
                "taxa_rejeicao_incorreta": (
                    falsas_rejeicoes / total_relacionadas
                    if total_relacionadas
                    else float("nan")
                ),
                "acuracia_intencoes_aceitas": (
                    classificadas_corretamente / aceitas_corretamente
                    if aceitas_corretamente
                    else float("nan")
                ),
            }
        )

    colunas = list(COLUNAS_THRESHOLD)
    tabela = pd.DataFrame(linhas)
    restantes = [coluna for coluna in tabela.columns if coluna not in colunas]
    return tabela[colunas + restantes]


# Critério de seleção, documentado e determinístico: maior Macro F1; em caso de
# empate, maior acurácia; persistindo o empate, o menor threshold, por ser a
# configuração menos restritiva (rejeitar uma dúvida legítima do jogador é mais
# custoso para a experiência do que aceitar uma pergunta fora de contexto).
CRITERIO_SELECAO_THRESHOLD = (
    "maior Macro F1; empate resolvido pela maior acuracia e, persistindo, "
    "pelo menor threshold"
)


def selecionar_threshold(tabela: pd.DataFrame) -> Dict[str, Any]:
    """Escolhe o threshold a partir da tabela de validação.

    Returns:
        Dicionário com ``threshold``, ``macro_f1``, ``acuracia``, ``criterio``
        e ``empatados`` (quantos limiares disputaram a primeira posição).
    """
    obrigatorias = {"threshold", "macro_f1", "acuracia"}
    faltantes = sorted(obrigatorias - set(tabela.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes na tabela de thresholds: {faltantes}")
    if tabela.empty:
        raise ValueError("A tabela de thresholds está vazia.")

    melhor_f1 = tabela["macro_f1"].max()
    candidatos = tabela.loc[np.isclose(tabela["macro_f1"], melhor_f1)]
    melhor_acuracia = candidatos["acuracia"].max()
    candidatos = candidatos.loc[np.isclose(candidatos["acuracia"], melhor_acuracia)]
    escolhido = candidatos.sort_values("threshold").iloc[0]

    return {
        "threshold": float(escolhido["threshold"]),
        "macro_f1": float(escolhido["macro_f1"]),
        "acuracia": float(escolhido["acuracia"]),
        "criterio": CRITERIO_SELECAO_THRESHOLD,
        "empatados": int(len(candidatos)),
    }


def imprimir_tabela_thresholds(tabela: pd.DataFrame, selecao: Mapping[str, Any]) -> None:
    """Imprime a grade de thresholds avaliada na validação e a escolha feita."""
    print("\n========================================")
    print("ANALISE DE THRESHOLDS (CONJUNTO DE VALIDACAO)")
    print("========================================")
    print(
        f"\n{'Thr.':>6}{'Acur.':>9}{'MacroF1':>10}{'AceitOK':>9}"
        f"{'RejOK':>7}{'FalsAce':>9}{'FalsRej':>9}{'ClassOK':>9}"
    )
    for linha in tabela.itertuples(index=False):
        print(
            f"{linha.threshold:>6.2f}{linha.acuracia:>9.4f}{linha.macro_f1:>10.4f}"
            f"{linha.aceitas_corretamente:>9d}{linha.rejeitadas_corretamente:>7d}"
            f"{linha.falsas_aceitacoes:>9d}{linha.falsas_rejeicoes:>9d}"
            f"{linha.classificadas_corretamente:>9d}"
        )
    print(f"\nCriterio de selecao: {selecao['criterio']}")
    print(f"Threshold selecionado: {selecao['threshold']:.2f}")
    print(f"Macro F1 na validacao: {selecao['macro_f1']:.4f}")
    print(f"Acuracia na validacao: {selecao['acuracia']:.4f}")
    if selecao["empatados"] > 1:
        print(f"Thresholds empatados no criterio: {selecao['empatados']}")


def distribuicao_similaridades(
    classificador: PuzzleClassifier,
    dataset: pd.DataFrame,
    matriz: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Maior similaridade frente às cinco intenções, por pergunta.

    Serve de base para o gráfico de distribuição e para inspecionar se existe
    separação entre perguntas relacionadas e não relacionadas.
    """
    _validar_dataset(dataset)
    frases = dataset["frase"].astype(str).tolist()
    if matriz is None:
        matriz = classificador.matriz_similaridade(frases)

    scores = classificador.similaridade_maxima_puzzle(matriz)
    tabela = dataset[["frase", "token_esperado"]].copy().reset_index(drop=True)
    tabela["similaridade_puzzle"] = scores
    tabela["relacionada"] = tabela["token_esperado"].isin(PUZZLE_TOKENS)
    return tabela


def _validar_dataset(dataset: pd.DataFrame) -> None:
    """Garante que o DataFrame possui as colunas mínimas e ao menos um registro."""
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("O dataset precisa ser um DataFrame do pandas.")
    faltantes = sorted({"frase", "token_esperado"} - set(dataset.columns))
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset: {faltantes}")
    if dataset.empty:
        raise ValueError("O dataset informado não possui registros.")


__all__ = [
    "CRITERIO_SELECAO_THRESHOLD",
    "DESCRICAO_CASOS",
    "ResultadoAvaliacao",
    "analisar_divergencias",
    "analisar_thresholds",
    "avaliar_classificador",
    "avaliar_embeddings",
    "avaliar_predicoes",
    "comparar_por_pergunta",
    "dificuldade_por_metodo",
    "distribuicao_similaridades",
    "imprimir_divergencias",
    "imprimir_relatorio",
    "imprimir_tabela_thresholds",
    "salvar_resultado",
    "selecionar_threshold",
    "tabela_comparativa",
]
