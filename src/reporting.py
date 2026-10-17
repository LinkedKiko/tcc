"""Geração dos relatórios em Markdown a partir de execuções reais.

Nada neste módulo cria números: todos os valores escritos vêm de objetos
:class:`~src.evaluation.ResultadoAvaliacao` e de tabelas produzidas pelos
experimentos. Seções sem dados são declaradas explicitamente como não
executadas, em vez de preenchidas com valores plausíveis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import PUZZLE_TOKENS, RESULTS_DIR
from .dataset import Particoes
from .evaluation import (
    DESCRICAO_CASOS,
    ResultadoAvaliacao,
    dificuldade_por_metodo,
    tabela_comparativa,
)

ARQUIVO_RELATORIO = "relatorio_semana2.md"
ARQUIVO_RESUMO_COMPARACAO = "comparacao_resumo.md"
ARQUIVO_COMPARACAO_CSV = "comparacao_classificadores.csv"
ARQUIVO_THRESHOLDS_CSV = "analise_thresholds.csv"
ARQUIVO_THRESHOLD_TESTE_CSV = "threshold_selecionado_teste.csv"

NAO_EXECUTADO = "_Esta análise não foi executada nesta rodada._"


@dataclass
class ContextoRelatorio:
    """Reúne tudo o que o relatório final precisa descrever."""

    particoes: Optional[Particoes] = None
    integridade: Optional[Mapping[str, Any]] = None
    resultado_keywords: Optional[ResultadoAvaliacao] = None
    resultado_embeddings: Optional[ResultadoAvaliacao] = None
    resultado_threshold: Optional[ResultadoAvaliacao] = None
    tabela_thresholds: Optional[pd.DataFrame] = None
    selecao_threshold: Optional[Mapping[str, Any]] = None
    comparacao: Optional[pd.DataFrame] = None
    divergencias: Optional[Mapping[str, pd.DataFrame]] = None
    distribuicao: Optional[pd.DataFrame] = None
    graficos: Mapping[str, Path] = field(default_factory=dict)
    arquivos: Mapping[str, Path] = field(default_factory=dict)

    @property
    def resultados_disponiveis(self) -> List[ResultadoAvaliacao]:
        return [
            resultado
            for resultado in (
                self.resultado_keywords,
                self.resultado_embeddings,
                self.resultado_threshold,
            )
            if resultado is not None
        ]


# ---------------------------------------------------------------------------
# Utilitários de formatação
# ---------------------------------------------------------------------------
def _celula(valor: Any, casas: int) -> str:
    """Formata um valor para a tabela Markdown sem depender de bibliotecas extras."""
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return "n/d"
    if isinstance(valor, (bool, np.bool_)):
        return "sim" if valor else "nao"
    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):.{casas}f}"
    return str(valor)


def _tabela_markdown(tabela: pd.DataFrame, casas: int = 4) -> str:
    """Converte um DataFrame em tabela Markdown com arredondamento uniforme."""
    if tabela is None or tabela.empty:
        return "_Tabela vazia._"

    cabecalho = [str(coluna) for coluna in tabela.columns]
    linhas = [
        [_celula(valor, casas) for valor in registro]
        for registro in tabela.itertuples(index=False, name=None)
    ]
    larguras = [
        max(len(cabecalho[indice]), *(len(linha[indice]) for linha in linhas))
        if linhas
        else len(cabecalho[indice])
        for indice in range(len(cabecalho))
    ]

    def formatar(celulas: Sequence[str]) -> str:
        preenchidas = [celula.ljust(larguras[i]) for i, celula in enumerate(celulas)]
        return "| " + " | ".join(preenchidas) + " |"

    separador = "| " + " | ".join("-" * largura for largura in larguras) + " |"
    return "\n".join([formatar(cabecalho), separador, *(formatar(linha) for linha in linhas)])


def _formatar_similaridade(valor: Any) -> str:
    if valor is None or pd.isna(valor):
        return "n/d"
    return f"{float(valor):.4f}"


def _linha_metricas(resultado: ResultadoAvaliacao) -> Dict[str, Any]:
    metricas = resultado.metricas
    return {
        "metodo": resultado.metodo,
        "conjunto": resultado.conjunto,
        "threshold": resultado.detalhes.get("threshold"),
        "total": metricas["total"],
        "acertos": metricas["acertos"],
        "erros": metricas["erros"],
        "acuracia": metricas["acuracia"],
        "macro_precision": metricas["macro_precision"],
        "macro_recall": metricas["macro_recall"],
        "macro_f1": metricas["macro_f1"],
    }


def _bloco_resultado(resultado: Optional[ResultadoAvaliacao]) -> str:
    """Resumo agregado, métricas por classe e matriz de confusão de um método."""
    if resultado is None:
        return NAO_EXECUTADO

    partes = [
        _tabela_markdown(pd.DataFrame([_linha_metricas(resultado)])),
        "",
        "Métricas por token:",
        "",
        _tabela_markdown(resultado.por_classe),
        "",
        "Matriz de confusão (linhas = token esperado, colunas = token previsto):",
        "",
        _tabela_markdown(resultado.matriz_confusao_df().reset_index(names="classe")),
        "",
        "Acurácia por nível de dificuldade:",
        "",
        _tabela_markdown(dificuldade_por_metodo(resultado)),
    ]
    return "\n".join(partes)


def _exemplos(tabela: pd.DataFrame, limite: int = 5) -> str:
    """Lista exemplos reais de uma seleção de perguntas."""
    if tabela.empty:
        return "_Nenhuma pergunta nesta categoria._"
    linhas = []
    for registro in tabela.head(limite).itertuples(index=False):
        similaridade = _formatar_similaridade(getattr(registro, "similaridade", None))
        esperado = getattr(registro, "token_esperado", "n/d")
        keywords = getattr(registro, "token_keywords", None)
        embeddings = getattr(registro, "token_embeddings", None)
        previsto = getattr(registro, "token_previsto", None)
        detalhe = []
        if keywords is not None:
            detalhe.append(f"keywords=`{keywords}`")
        if embeddings is not None:
            detalhe.append(f"embeddings=`{embeddings}`")
        if previsto is not None:
            detalhe.append(f"previsto=`{previsto}`")
        linhas.append(
            f"- \"{registro.frase}\"  \n"
            f"  esperado=`{esperado}` | " + " | ".join(detalhe) + f" | similaridade={similaridade}"
        )
    if len(tabela) > limite:
        linhas.append(f"- _(+{len(tabela) - limite} pergunta(s) nesta categoria)_")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Seções do relatório
# ---------------------------------------------------------------------------
def _secao_dataset(contexto: ContextoRelatorio) -> str:
    if contexto.particoes is None:
        return NAO_EXECUTADO

    particoes = contexto.particoes
    partes = [
        f"Divisão estratificada determinística com `RANDOM_STATE = {particoes.random_state}` "
        f"e {particoes.validation_size:.0%} das frases para validação.",
        "",
        "Perguntas por token e por partição:",
        "",
        _tabela_markdown(particoes.resumo()),
        "",
        "Distribuição por nível de dificuldade no conjunto completo:",
        "",
        _tabela_markdown(
            pd.crosstab(
                particoes.completo["token_esperado"], particoes.completo["dificuldade"]
            ).reset_index()
        ),
    ]

    if contexto.integridade:
        integridade = contexto.integridade
        comuns = integridade.get("frases_em_comum", [])
        sobreposicao = integridade.get("sobreposicao_particoes", [])
        partes += [
            "",
            "Verificações de integridade:",
            "",
            f"- Frases de referência: {integridade.get('total_referencia')}",
            f"- Frases de avaliação: {integridade.get('total_avaliacao')}",
            f"- Frases idênticas entre referência e avaliação: {len(comuns)}",
            f"- Frases presentes em validação e teste ao mesmo tempo: {len(sobreposicao)}",
        ]
    return "\n".join(partes)


def _secao_thresholds(contexto: ContextoRelatorio) -> str:
    if contexto.tabela_thresholds is None or contexto.tabela_thresholds.empty:
        return NAO_EXECUTADO

    colunas = [
        "threshold",
        "acuracia",
        "macro_f1",
        "aceitas_corretamente",
        "rejeitadas_corretamente",
        "falsas_aceitacoes",
        "falsas_rejeicoes",
        "classificadas_corretamente",
        "taxa_rejeicao_correta",
        "taxa_rejeicao_incorreta",
    ]
    presentes = [coluna for coluna in colunas if coluna in contexto.tabela_thresholds.columns]
    partes = [
        "Resultados obtidos **exclusivamente no conjunto de validação**. "
        "O conjunto de teste não participou da escolha do limiar.",
        "",
        _tabela_markdown(contexto.tabela_thresholds[presentes]),
    ]
    if contexto.selecao_threshold:
        selecao = contexto.selecao_threshold
        partes += [
            "",
            f"- Critério de seleção: {selecao['criterio']}",
            f"- Threshold selecionado: **{selecao['threshold']:.2f}**",
            f"- Macro F1 na validação: {selecao['macro_f1']:.4f}",
            f"- Acurácia na validação: {selecao['acuracia']:.4f}",
            f"- Limiares empatados no critério: {selecao['empatados']}",
        ]
    if contexto.distribuicao is not None and not contexto.distribuicao.empty:
        agrupado = (
            contexto.distribuicao.groupby("relacionada")["similaridade_puzzle"]
            .agg(["count", "mean", "min", "max"])
            .rename(index={True: "relacionadas", False: "nao_relacionadas"})
            .reset_index()
            .rename(columns={"relacionada": "grupo"})
        )
        partes += [
            "",
            "Similaridade máxima frente às cinco intenções do puzzle:",
            "",
            _tabela_markdown(agrupado),
        ]
    return "\n".join(partes)


def _secao_comparacao(contexto: ContextoRelatorio) -> str:
    resultados = contexto.resultados_disponiveis
    if not resultados:
        return NAO_EXECUTADO
    return _tabela_markdown(tabela_comparativa(resultados))


def _secao_erros(contexto: ContextoRelatorio) -> str:
    partes: List[str] = []

    if contexto.divergencias:
        for chave, descricao in DESCRICAO_CASOS.items():
            if chave == "caso_c":
                # Concordância correta não é um erro; aparece apenas na contagem.
                bloco = contexto.divergencias.get(chave, pd.DataFrame())
                partes += [f"### {descricao}", "", f"Total: {len(bloco)} pergunta(s).", ""]
                continue
            bloco = contexto.divergencias.get(chave, pd.DataFrame())
            partes += [
                f"### {descricao}",
                "",
                f"Total: {len(bloco)} pergunta(s).",
                "",
                _exemplos(bloco),
                "",
            ]
    else:
        partes += ["### Divergências entre métodos", "", NAO_EXECUTADO, ""]

    resultado_threshold = contexto.resultado_threshold
    if resultado_threshold is not None:
        registros = resultado_threshold.registros
        relacionadas = registros["token_esperado"].isin(PUZZLE_TOKENS)
        aceitas_indevidas = registros.loc[
            ~relacionadas & (registros["token_previsto"] != "TOKEN_NAO_RELACIONADO")
        ]
        rejeitadas_indevidas = registros.loc[
            relacionadas & (registros["token_previsto"] == "TOKEN_NAO_RELACIONADO")
        ]
        partes += [
            "### Perguntas não relacionadas aceitas incorretamente (modo de rejeição)",
            "",
            f"Total: {len(aceitas_indevidas)} pergunta(s).",
            "",
            _exemplos(aceitas_indevidas),
            "",
            "### Perguntas relacionadas rejeitadas incorretamente (modo de rejeição)",
            "",
            f"Total: {len(rejeitadas_indevidas)} pergunta(s).",
            "",
            _exemplos(rejeitadas_indevidas),
            "",
        ]
    else:
        partes += [
            "### Aceitações e rejeições incorretas do modo de rejeição",
            "",
            NAO_EXECUTADO,
            "",
        ]
    return "\n".join(partes).rstrip()


SECAO_LIMITACOES = """\
As observações abaixo delimitam o alcance dos números apresentados.

- **Dimensão reduzida do dataset.** São 90 frases de referência e 100 perguntas
  de avaliação. Com 6 perguntas por classe na validação e 9 no teste (25 e 15
  para `TOKEN_NAO_RELACIONADO`), cada acerto ou erro desloca as métricas em
  vários pontos percentuais. Os intervalos de confiança são largos e as
  diferenças pequenas entre métodos não devem ser lidas como conclusivas.
- **Seleção manual das perguntas.** As perguntas de avaliação e seus rótulos
  foram elaborados pelo próprio autor, a partir da descrição do puzzle. Não
  houve anotação independente nem medição de concordância entre avaliadores,
  portanto os rótulos devem ser tratados como uma proposta sujeita a revisão
  humana, não como padrão-ouro validado.
- **Possível ambiguidade semântica.** Algumas intenções do puzzle se sobrepõem
  por construção: o relógio abre o cofre, o cofre completa o mural e o mural dá
  valor aos símbolos. Perguntas que citam dois elementos admitem mais de uma
  leitura plausível, e o rótulo escolhido reflete a intenção principal julgada
  pelo autor.
- **Influência das frases de referência.** O classificador semântico só
  consegue reconhecer uma intenção se houver uma frase de referência próxima.
  O desempenho medido é tanto uma avaliação do modelo de embeddings quanto do
  conjunto de 90 frases escolhido na Semana 1.
- **Sensibilidade do threshold.** O limiar foi escolhido em uma grade
  exploratória de nove valores, usando apenas 40 perguntas de validação. Um
  conjunto de validação diferente poderia indicar outro valor, e os limiares
  observados são específicos deste modelo de embeddings.
- **Ausência de avaliação com jogadores reais.** Nenhuma pergunta foi coletada
  de pessoas jogando o puzzle. A formulação espontânea de dúvidas durante o
  jogo pode diferir das perguntas escritas em ambiente controlado.
- **Baseline definido a priori.** As listas de palavras-chave foram montadas a
  partir da descrição do puzzle e do vocabulário das frases de referência, sem
  consultar o conjunto de avaliação. Um baseline ajustado iterativamente aos
  dados teria desempenho diferente.
"""


# ---------------------------------------------------------------------------
# Relatórios
# ---------------------------------------------------------------------------
def gerar_relatorio_semana2(
    contexto: ContextoRelatorio,
    diretorio: Path | str = RESULTS_DIR,
) -> Path:
    """Escreve ``results/relatorio_semana2.md`` com os resultados da rodada."""
    destino = Path(diretorio)
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / ARQUIVO_RELATORIO

    agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    secoes = [
        "# Relatório da Semana 2",
        "",
        "Documento gerado automaticamente por `evaluate.py`. Todos os números "
        "vêm da execução real dos classificadores sobre os arquivos em `data/`.",
        "",
        f"- Gerado em: {agora}",
        "",
        "## 1. Dataset",
        "",
        _secao_dataset(contexto),
        "",
        "## 2. Classificador por palavras-chave",
        "",
        _bloco_resultado(contexto.resultado_keywords),
        "",
        "## 3. Classificador por embeddings (sem threshold)",
        "",
        _bloco_resultado(contexto.resultado_embeddings),
        "",
        "## 4. Análise de thresholds",
        "",
        _secao_thresholds(contexto),
        "",
        "## 5. Avaliação com o threshold selecionado",
        "",
        _bloco_resultado(contexto.resultado_threshold),
        "",
        "## 6. Comparação dos cenários",
        "",
        _secao_comparacao(contexto),
        "",
        "## 7. Análise de erros",
        "",
        _secao_erros(contexto),
        "",
        "## 8. Limitações",
        "",
        SECAO_LIMITACOES,
        "",
        "## 9. Arquivos gerados",
        "",
        _secao_arquivos(contexto),
        "",
    ]
    caminho.write_text("\n".join(secoes), encoding="utf-8")
    return caminho


def _secao_arquivos(contexto: ContextoRelatorio) -> str:
    itens = list(contexto.arquivos.items()) + list(contexto.graficos.items())
    if not itens:
        return "_Nenhum arquivo adicional registrado nesta rodada._"
    linhas = []
    for nome, caminho in itens:
        caminho = Path(caminho)
        try:
            exibicao = caminho.relative_to(RESULTS_DIR.parent).as_posix()
        except ValueError:
            exibicao = caminho.as_posix()
        linhas.append(f"- `{exibicao}` ({nome})")
    return "\n".join(sorted(linhas))


def gerar_resumo_comparacao(
    resultados: Sequence[ResultadoAvaliacao],
    divergencias: Optional[Mapping[str, pd.DataFrame]] = None,
    diretorio: Path | str = RESULTS_DIR,
) -> Path:
    """Escreve ``results/comparacao_resumo.md`` com a leitura da comparação."""
    if not resultados:
        raise ValueError("É necessário informar ao menos um resultado.")

    destino = Path(diretorio)
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / ARQUIVO_RESUMO_COMPARACAO

    tabela = tabela_comparativa(resultados)
    melhor_acuracia = tabela.loc[tabela["acuracia"].idxmax()]
    melhor_f1 = tabela.loc[tabela["macro_f1"].idxmax()]

    partes = [
        "# Comparação dos classificadores",
        "",
        f"Conjunto avaliado: **{resultados[0].conjunto}** "
        f"({resultados[0].metricas['total']} perguntas, idênticas para todos os métodos).",
        "",
        "## Métricas agregadas",
        "",
        _tabela_markdown(tabela),
        "",
        "## Leitura dos resultados",
        "",
        f"- Maior acurácia: **{melhor_acuracia['metodo']}** "
        f"({melhor_acuracia['acuracia']:.2%}).",
        f"- Maior Macro F1: **{melhor_f1['metodo']}** ({melhor_f1['macro_f1']:.4f}).",
        "",
        "## Dificuldade por token (F1 por classe)",
        "",
        _tabela_markdown(_f1_por_classe(resultados)),
        "",
        "## Dificuldade por nível das perguntas",
        "",
        _tabela_markdown(
            pd.concat([dificuldade_por_metodo(r) for r in resultados], ignore_index=True)
        ),
        "",
    ]

    if divergencias:
        partes += ["## Divergências pergunta a pergunta", ""]
        for chave, descricao in DESCRICAO_CASOS.items():
            bloco = divergencias.get(chave, pd.DataFrame())
            partes += [
                f"### {descricao} ({len(bloco)} pergunta(s))",
                "",
                _exemplos(bloco),
                "",
            ]

    caminho.write_text("\n".join(partes), encoding="utf-8")
    return caminho


def _f1_por_classe(resultados: Sequence[ResultadoAvaliacao]) -> pd.DataFrame:
    """Tabela com o F1 de cada token lado a lado para todos os métodos."""
    tabela = None
    for resultado in resultados:
        coluna = resultado.por_classe[["token", "f1"]].rename(
            columns={"f1": f"f1_{resultado.metodo}"}
        )
        tabela = coluna if tabela is None else tabela.merge(coluna, on="token")
    suporte = resultados[0].por_classe[["token", "suporte"]]
    return tabela.merge(suporte, on="token")


__all__ = [
    "ARQUIVO_COMPARACAO_CSV",
    "ARQUIVO_RELATORIO",
    "ARQUIVO_RESUMO_COMPARACAO",
    "ARQUIVO_THRESHOLDS_CSV",
    "ARQUIVO_THRESHOLD_TESTE_CSV",
    "ContextoRelatorio",
    "gerar_relatorio_semana2",
    "gerar_resumo_comparacao",
]
