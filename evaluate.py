"""Script de avaliação experimental da Semana 2.

Modos disponíveis::

    python evaluate.py --modo embeddings   # métricas do classificador semântico
    python evaluate.py --modo baseline     # métricas do baseline por palavras-chave
    python evaluate.py --modo comparar     # os dois métodos sobre as mesmas perguntas
    python evaluate.py --modo thresholds   # grade de limiares sobre a validação
    python evaluate.py --modo completo     # tudo acima, com gráficos e relatórios

O modelo de embeddings é carregado no máximo uma vez por execução e a matriz de
similaridades de cada partição é calculada uma única vez e reaproveitada por
todos os experimentos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.classifier import PuzzleClassifier
from src.config import (
    CONJUNTO_TESTE,
    CONJUNTO_VALIDACAO,
    DATASET_PATH,
    MANUAL_TEST_PATH,
    MODO_PADRAO,
    MODO_REJEICAO,
    RANDOM_STATE,
    RESULTS_DIR,
    THRESHOLDS,
    VALIDATION_SIZE,
)
from src.dataset import (
    Particoes,
    preparar_particoes,
    salvar_particoes,
    verificar_integridade,
)
from src.evaluation import (
    ResultadoAvaliacao,
    analisar_divergencias,
    analisar_thresholds,
    avaliar_classificador,
    avaliar_embeddings,
    comparar_por_pergunta,
    distribuicao_similaridades,
    imprimir_divergencias,
    imprimir_relatorio,
    imprimir_tabela_thresholds,
    salvar_resultado,
    selecionar_threshold,
    tabela_comparativa,
)
from src.keyword_classifier import KeywordClassifier
from src.reporting import (
    ARQUIVO_COMPARACAO_CSV,
    ARQUIVO_THRESHOLD_TESTE_CSV,
    ARQUIVO_THRESHOLDS_CSV,
    ContextoRelatorio,
    gerar_relatorio_semana2,
    gerar_resumo_comparacao,
)

METODO_KEYWORDS = "Palavras-chave"
METODO_EMBEDDINGS = "Embeddings sem threshold"


class Experimento:
    """Mantém os dados, os classificadores e os embeddings reutilizados.

    A classe existe para evitar recarregar o modelo e recalcular embeddings
    entre os experimentos de uma mesma execução.
    """

    def __init__(
        self,
        caminho_referencia: Path = DATASET_PATH,
        caminho_avaliacao: Path = MANUAL_TEST_PATH,
        validation_size: float = VALIDATION_SIZE,
        random_state: int = RANDOM_STATE,
        diretorio_resultados: Path = RESULTS_DIR,
    ) -> None:
        self.caminho_referencia = Path(caminho_referencia)
        self.diretorio_resultados = Path(diretorio_resultados)
        self.particoes: Particoes = preparar_particoes(
            caminho_avaliacao, validation_size=validation_size, random_state=random_state
        )
        self.integridade = verificar_integridade(self.particoes)
        self.keyword_classifier = KeywordClassifier()
        self._classificador: Optional[PuzzleClassifier] = None
        self._matrizes: Dict[str, np.ndarray] = {}
        self.arquivos: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    @property
    def classificador(self) -> PuzzleClassifier:
        """Carrega o modelo de embeddings sob demanda, uma única vez."""
        if self._classificador is None:
            print("Carregando o modelo de embeddings e as frases de referência...")
            self._classificador = PuzzleClassifier(self.caminho_referencia)
        return self._classificador

    def dataset(self, conjunto: str) -> pd.DataFrame:
        return self.particoes.obter(conjunto)

    def matriz(self, conjunto: str) -> np.ndarray:
        """Matriz de similaridades da partição, calculada uma única vez."""
        if conjunto not in self._matrizes:
            dados = self.dataset(conjunto)
            self._matrizes[conjunto] = self.classificador.matriz_similaridade(
                dados["frase"].astype(str).tolist()
            )
        return self._matrizes[conjunto]

    # ------------------------------------------------------------------
    def registrar_particoes(self) -> None:
        """Salva as partições em ``results/`` e imprime o resumo."""
        caminhos = salvar_particoes(self.particoes, self.diretorio_resultados)
        self.arquivos.update({f"particao_{nome}": caminho for nome, caminho in caminhos.items()})

        print("\n========================================")
        print("DIVISAO DETERMINISTICA DO CONJUNTO DE AVALIACAO")
        print("========================================")
        print(
            f"\nRANDOM_STATE = {self.particoes.random_state} | "
            f"validacao = {self.particoes.validation_size:.0%}"
        )
        print()
        print(self.particoes.resumo().to_string(index=False))

        integridade = self.integridade
        print("\nVerificacoes de integridade:")
        print(f"  Frases de referencia: {integridade['total_referencia']}")
        print(f"  Frases de avaliacao: {integridade['total_avaliacao']}")
        print(f"  Frases identicas entre referencia e avaliacao: {len(integridade['frases_em_comum'])}")
        print(
            "  Frases em validacao e teste ao mesmo tempo: "
            f"{len(integridade['sobreposicao_particoes'])}"
        )

    # ------------------------------------------------------------------
    def avaliar_baseline(self, conjunto: str = CONJUNTO_TESTE) -> ResultadoAvaliacao:
        return avaliar_classificador(
            self.keyword_classifier,
            self.dataset(conjunto),
            metodo=METODO_KEYWORDS,
            conjunto=conjunto,
        )

    def avaliar_embeddings_sem_threshold(
        self, conjunto: str = CONJUNTO_TESTE
    ) -> ResultadoAvaliacao:
        return avaliar_embeddings(
            self.classificador,
            self.dataset(conjunto),
            conjunto=conjunto,
            modo=MODO_PADRAO,
            threshold=None,
            matriz=self.matriz(conjunto),
            metodo=METODO_EMBEDDINGS,
        )

    def avaliar_embeddings_com_threshold(
        self, threshold: float, conjunto: str = CONJUNTO_TESTE
    ) -> ResultadoAvaliacao:
        return avaliar_embeddings(
            self.classificador,
            self.dataset(conjunto),
            conjunto=conjunto,
            modo=MODO_REJEICAO,
            threshold=threshold,
            matriz=self.matriz(conjunto),
            metodo=f"Embeddings com threshold {threshold:.2f}",
        )

    def analisar_thresholds(self, thresholds=THRESHOLDS) -> pd.DataFrame:
        """Grade de limiares avaliada somente no conjunto de validação."""
        return analisar_thresholds(
            self.classificador,
            self.dataset(CONJUNTO_VALIDACAO),
            thresholds=thresholds,
            matriz=self.matriz(CONJUNTO_VALIDACAO),
        )

    def salvar(self, resultado: ResultadoAvaliacao, prefixo: str) -> None:
        caminhos = salvar_resultado(resultado, prefixo, self.diretorio_resultados)
        self.arquivos.update({f"{prefixo}_{nome}": caminho for nome, caminho in caminhos.items()})
        print(f"\nResultados salvos em: {caminhos['detalhado'].parent}")
        for nome, caminho in caminhos.items():
            print(f"  {nome}: {caminho.name}")


# ---------------------------------------------------------------------------
# Modos de execução
# ---------------------------------------------------------------------------
def modo_embeddings(experimento: Experimento, conjunto: str) -> ResultadoAvaliacao:
    resultado = experimento.avaliar_embeddings_sem_threshold(conjunto)
    imprimir_relatorio(resultado)
    experimento.salvar(resultado, f"embeddings_{conjunto}")
    return resultado


def modo_baseline(experimento: Experimento, conjunto: str) -> ResultadoAvaliacao:
    resultado = experimento.avaliar_baseline(conjunto)
    imprimir_relatorio(resultado)
    experimento.salvar(resultado, f"baseline_{conjunto}")
    return resultado


def modo_comparar(
    experimento: Experimento,
    conjunto: str,
    resultados_extra: Optional[List[ResultadoAvaliacao]] = None,
    salvar_individuais: bool = True,
) -> Dict[str, object]:
    """Executa os dois métodos sobre exatamente as mesmas perguntas."""
    resultado_keywords = experimento.avaliar_baseline(conjunto)
    resultado_embeddings = experimento.avaliar_embeddings_sem_threshold(conjunto)

    if salvar_individuais:
        experimento.salvar(resultado_keywords, f"baseline_{conjunto}")
        experimento.salvar(resultado_embeddings, f"embeddings_{conjunto}")

    comparacao = comparar_por_pergunta(resultado_keywords, resultado_embeddings)
    divergencias = analisar_divergencias(comparacao)

    todos = [resultado_keywords, resultado_embeddings] + list(resultados_extra or [])
    tabela = tabela_comparativa(todos)

    print("\n========================================")
    print("COMPARACAO DOS CLASSIFICADORES")
    print("========================================")
    print(f"\nConjunto: {conjunto} | perguntas: {resultado_keywords.metricas['total']}")
    print("\n" + tabela.to_string(index=False))
    imprimir_divergencias(divergencias)

    destino = experimento.diretorio_resultados
    destino.mkdir(parents=True, exist_ok=True)
    caminho_comparacao = destino / ARQUIVO_COMPARACAO_CSV
    comparacao.to_csv(caminho_comparacao, index=False, encoding="utf-8")
    caminho_tabela = destino / "comparacao_metricas.csv"
    tabela.to_csv(caminho_tabela, index=False, encoding="utf-8")
    caminho_resumo = gerar_resumo_comparacao(todos, divergencias, destino)

    experimento.arquivos.update(
        {
            "comparacao_por_pergunta": caminho_comparacao,
            "comparacao_metricas": caminho_tabela,
            "comparacao_resumo": caminho_resumo,
        }
    )
    print(f"\nComparacao salva em: {caminho_comparacao}")
    print(f"Resumo salvo em: {caminho_resumo}")

    return {
        "keywords": resultado_keywords,
        "embeddings": resultado_embeddings,
        "comparacao": comparacao,
        "divergencias": divergencias,
        "tabela": tabela,
    }


def modo_thresholds(experimento: Experimento, thresholds=THRESHOLDS) -> Dict[str, object]:
    """Analisa a grade de limiares usando apenas o conjunto de validação."""
    tabela = experimento.analisar_thresholds(thresholds)
    selecao = selecionar_threshold(tabela)
    imprimir_tabela_thresholds(tabela, selecao)

    distribuicao = distribuicao_similaridades(
        experimento.classificador,
        experimento.dataset(CONJUNTO_VALIDACAO),
        experimento.matriz(CONJUNTO_VALIDACAO),
    )
    agrupado = distribuicao.groupby("relacionada")["similaridade_puzzle"].agg(
        ["count", "mean", "min", "max"]
    )
    print("\nSimilaridade maxima frente as cinco intencoes (validacao):")
    print(agrupado.to_string())

    destino = experimento.diretorio_resultados
    destino.mkdir(parents=True, exist_ok=True)
    caminho_tabela = destino / ARQUIVO_THRESHOLDS_CSV
    tabela.to_csv(caminho_tabela, index=False, encoding="utf-8")
    caminho_selecao = destino / "threshold_selecionado.csv"
    pd.DataFrame([selecao]).to_csv(caminho_selecao, index=False, encoding="utf-8")
    caminho_distribuicao = destino / "distribuicao_similaridades_validacao.csv"
    distribuicao.to_csv(caminho_distribuicao, index=False, encoding="utf-8")

    experimento.arquivos.update(
        {
            "analise_thresholds": caminho_tabela,
            "threshold_selecionado": caminho_selecao,
            "distribuicao_similaridades": caminho_distribuicao,
        }
    )
    print(f"\nTabela de thresholds salva em: {caminho_tabela}")
    print(f"Threshold selecionado registrado em: {caminho_selecao}")

    return {"tabela": tabela, "selecao": selecao, "distribuicao": distribuicao}


def modo_completo(
    experimento: Experimento,
    thresholds=THRESHOLDS,
    gerar_graficos: bool = True,
) -> ContextoRelatorio:
    """Executa a sequência completa de experimentos da Semana 2."""
    experimento.registrar_particoes()

    # 1) Seleção do limiar usando exclusivamente a validação.
    analise = modo_thresholds(experimento, thresholds)
    threshold = float(analise["selecao"]["threshold"])

    # 2) Avaliação única no conjunto de teste com o limiar já escolhido.
    comparacao = modo_comparar(experimento, CONJUNTO_TESTE, salvar_individuais=True)
    resultado_threshold = experimento.avaliar_embeddings_com_threshold(
        threshold, CONJUNTO_TESTE
    )
    imprimir_relatorio(resultado_threshold)
    experimento.salvar(resultado_threshold, f"embeddings_threshold_{CONJUNTO_TESTE}")

    destino = experimento.diretorio_resultados
    caminho_threshold_teste = destino / ARQUIVO_THRESHOLD_TESTE_CSV
    pd.DataFrame(
        [
            {
                "threshold_selecionado": threshold,
                "criterio": analise["selecao"]["criterio"],
                "macro_f1_validacao": analise["selecao"]["macro_f1"],
                "acuracia_validacao": analise["selecao"]["acuracia"],
                **{
                    f"{chave}_teste": valor
                    for chave, valor in resultado_threshold.metricas.items()
                },
            }
        ]
    ).to_csv(caminho_threshold_teste, index=False, encoding="utf-8")
    experimento.arquivos["threshold_avaliado_no_teste"] = caminho_threshold_teste

    resultados_finais = [
        comparacao["keywords"],
        comparacao["embeddings"],
        resultado_threshold,
    ]
    tabela_final = tabela_comparativa(resultados_finais)
    caminho_tabela_final = destino / "comparacao_metricas.csv"
    tabela_final.to_csv(caminho_tabela_final, index=False, encoding="utf-8")
    experimento.arquivos["comparacao_metricas"] = caminho_tabela_final

    print("\n========================================")
    print("COMPARACAO FINAL DOS TRES CENARIOS (CONJUNTO DE TESTE)")
    print("========================================")
    print("\n" + tabela_final.to_string(index=False))

    distribuicao_teste = distribuicao_similaridades(
        experimento.classificador,
        experimento.dataset(CONJUNTO_TESTE),
        experimento.matriz(CONJUNTO_TESTE),
    )

    graficos: Dict[str, Path] = {}
    if gerar_graficos:
        from src import plots

        graficos = plots.gerar_todos(
            resultados={
                "palavras_chave": comparacao["keywords"],
                "embeddings_sem_threshold": comparacao["embeddings"],
                "embeddings_com_threshold": resultado_threshold,
            },
            tabela_thresholds=analise["tabela"],
            distribuicao=analise["distribuicao"],
            threshold_selecionado=threshold,
            diretorio=destino,
        )
        print("\nGraficos gerados:")
        for nome, caminho in graficos.items():
            print(f"  {nome}: {caminho.name}")

    contexto = ContextoRelatorio(
        particoes=experimento.particoes,
        integridade=experimento.integridade,
        resultado_keywords=comparacao["keywords"],
        resultado_embeddings=comparacao["embeddings"],
        resultado_threshold=resultado_threshold,
        tabela_thresholds=analise["tabela"],
        selecao_threshold=analise["selecao"],
        comparacao=comparacao["comparacao"],
        divergencias=comparacao["divergencias"],
        distribuicao=distribuicao_teste,
        graficos=graficos,
        arquivos=experimento.arquivos,
    )
    caminho_relatorio = gerar_relatorio_semana2(contexto, destino)
    print(f"\nRelatorio final: {caminho_relatorio}")

    print("\n========================================")
    print("RESUMO DA EXECUCAO")
    print("========================================")
    print(f"\nThreshold selecionado na validacao: {threshold:.2f}")
    print(f"Criterio: {analise['selecao']['criterio']}")
    print(f"Perguntas no conjunto de teste: {resultado_threshold.metricas['total']}")
    for resultado in resultados_finais:
        print(
            f"  {resultado.metodo}: acuracia={resultado.metricas['acuracia']:.2%} "
            f"| macro F1={resultado.metricas['macro_f1']:.4f}"
        )
    return contexto


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimentos de avaliação da Semana 2 do TCC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modo",
        required=True,
        choices=("embeddings", "baseline", "comparar", "thresholds", "completo"),
        help="Análise a executar.",
    )
    parser.add_argument(
        "--conjunto",
        default=CONJUNTO_TESTE,
        choices=(CONJUNTO_VALIDACAO, CONJUNTO_TESTE, "completo"),
        help="Partição usada nos modos embeddings, baseline e comparar.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(THRESHOLDS),
        help="Grade de limiares da análise de rejeição.",
    )
    parser.add_argument(
        "--sem-graficos",
        action="store_true",
        help="Desativa a geração de PNGs no modo completo.",
    )
    parser.add_argument(
        "--referencia",
        type=Path,
        default=DATASET_PATH,
        help="CSV de frases de referência.",
    )
    parser.add_argument(
        "--avaliacao",
        type=Path,
        default=MANUAL_TEST_PATH,
        help="CSV de perguntas de avaliação.",
    )
    parser.add_argument(
        "--resultados",
        type=Path,
        default=RESULTS_DIR,
        help="Diretório onde os relatórios são gravados.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = construir_parser().parse_args(argv)

    try:
        experimento = Experimento(
            caminho_referencia=args.referencia,
            caminho_avaliacao=args.avaliacao,
            diretorio_resultados=args.resultados,
        )

        if args.modo == "completo":
            modo_completo(
                experimento,
                thresholds=args.thresholds,
                gerar_graficos=not args.sem_graficos,
            )
        elif args.modo == "thresholds":
            experimento.registrar_particoes()
            modo_thresholds(experimento, args.thresholds)
        elif args.modo == "comparar":
            experimento.registrar_particoes()
            modo_comparar(experimento, args.conjunto)
        elif args.modo == "baseline":
            modo_baseline(experimento, args.conjunto)
        else:
            modo_embeddings(experimento, args.conjunto)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Erro ao executar a avaliacao: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
