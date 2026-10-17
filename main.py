"""Ponto de entrada do protótipo executado pelo terminal.

Uso::

    python main.py                       # modo interativo
    python main.py --test                # roda data/teste_manual.csv inteiro
    python main.py --threshold 0.55      # ativa o modo experimental de rejeição

As análises experimentais da Semana 2 ficam em ``evaluate.py``. Este arquivo
preserva a demonstração interativa da Semana 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.classifier import PuzzleClassifier
from src.config import (
    CONJUNTO_COMPLETO,
    DATASET_PATH,
    MANUAL_TEST_PATH,
    MODO_PADRAO,
    MODO_REJEICAO,
    TOP_K,
)
from src.dataset import carregar_avaliacao
from src.evaluation import avaliar_classificador, imprimir_relatorio
from src.responses import obter_dica


class ClassificadorDemonstracao:
    """Adaptador que fixa modo e limiar, expondo a interface de avaliação."""

    def __init__(
        self,
        classificador: PuzzleClassifier,
        modo: str = MODO_PADRAO,
        threshold: Optional[float] = None,
    ) -> None:
        self.classificador = classificador
        self.modo = modo
        self.threshold = threshold

    @property
    def descricao(self) -> str:
        if self.modo == MODO_REJEICAO:
            return f"Embeddings com rejeicao (threshold {self.threshold:.2f})"
        return "Embeddings sem threshold"

    def classificar(self, pergunta: str) -> Dict[str, Any]:
        return self.classificador.classificar(pergunta, self.modo, self.threshold)

    def classificar_em_lote(self, perguntas) -> list[Dict[str, Any]]:
        return self.classificador.classificar_em_lote(perguntas, self.modo, self.threshold)


def exibir_resultado(resultado: Dict[str, Any]) -> None:
    """Exibe uma classificação no formato definido para a demonstração."""
    print("\n----------------------------------------")
    print(f"Pergunta:\n{resultado['pergunta']}")
    print("\nFrase de referência mais próxima:")
    print(resultado["frase_referencia"])
    print(f"\nToken identificado:\n{resultado['token']}")
    print(f"\nSimilaridade:\n{resultado['similaridade']:.4f}")
    if resultado.get("rejeitada"):
        print(
            f"\nA pergunta foi rejeitada: a similaridade ficou abaixo do limiar "
            f"de {resultado['threshold']:.2f}."
        )
        print(f"Intenção mais próxima descartada: {resultado['token_intencao']}")
    print(f"\nTop {len(resultado['top_resultados'])} resultados:")
    for posicao, item in enumerate(resultado["top_resultados"], start=1):
        print(f"\n{posicao}. {item['token']} | {item['similaridade']:.4f}")
        print(f"   {item['frase']}")
    print(f"\nDica:\n{obter_dica(resultado['token'])}")
    print("----------------------------------------")


def executar_teste_manual(
    classificador: ClassificadorDemonstracao,
    caminho: Path = MANUAL_TEST_PATH,
) -> None:
    """Executa todo o conjunto manual e imprime o relatório de avaliação.

    Reutiliza o módulo de avaliação, portanto as métricas são as mesmas
    produzidas por ``evaluate.py`` quando aplicadas ao conjunto completo.
    """
    dataset = carregar_avaliacao(caminho)
    resultado = avaliar_classificador(
        classificador,
        dataset,
        metodo=classificador.descricao,
        conjunto=CONJUNTO_COMPLETO,
    )
    imprimir_relatorio(resultado)
    print(
        "\nObservacao: este conjunto reune validacao e teste. Para a avaliacao "
        "experimental com particoes separadas, use evaluate.py."
    )


def executar_interativo(classificador: ClassificadorDemonstracao) -> None:
    """Executa o modo interativo até o comando sair ou EOF."""
    print("========================================")
    print("     SISTEMA DE DICAS - PROTÓTIPO")
    print("========================================")
    print(f"Modo: {classificador.descricao}")
    print("Digite 'sair' para encerrar.")

    while True:
        try:
            pergunta = input("\nDigite sua pergunta:\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if pergunta.strip().casefold() == "sair":
            print("Encerrando.")
            break
        if not pergunta.strip():
            print("A pergunta não pode ser vazia.")
            continue

        exibir_resultado(classificador.classificar(pergunta))


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Protótipo de dicas por embeddings.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executa as perguntas de data/teste_manual.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Ativa o modo experimental de rejeição com o limiar informado. "
            "Sem este argumento, vale a maior similaridade (comportamento padrão)."
        ),
    )
    parser.add_argument(
        "--referencia",
        type=Path,
        default=DATASET_PATH,
        help="CSV de frases de referência.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Quantidade de frases de referência exibidas no ranking.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Executa o modo interativo ou o teste manual conforme os argumentos."""
    args = construir_parser().parse_args(argv)

    try:
        modo = MODO_REJEICAO if args.threshold is not None else MODO_PADRAO
        classificador = ClassificadorDemonstracao(
            PuzzleClassifier(args.referencia, top_k=args.top_k),
            modo=modo,
            threshold=args.threshold,
        )
        if args.test:
            executar_teste_manual(classificador)
        else:
            executar_interativo(classificador)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Erro ao iniciar o protótipo: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
