"""Ponto de entrada do protótipo executado pelo terminal."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import pandas as pd

from src.classifier import PuzzleClassifier
from src.config import DATASET_PATH, MANUAL_TEST_PATH, TOP_K
from src.responses import obter_dica


def exibir_resultado(resultado: Dict[str, Any]) -> None:
    """Exibe uma classificação no formato definido para a demonstração."""
    print("\n----------------------------------------")
    print(f"Pergunta:\n{resultado['pergunta']}")
    print("\nFrase de referência mais próxima:")
    print(resultado["frase_referencia"])
    print(f"\nToken identificado:\n{resultado['token']}")
    print(f"\nSimilaridade:\n{resultado['similaridade']:.4f}")
    print(f"\nTop {TOP_K} resultados:")
    for posicao, item in enumerate(resultado["top_resultados"], start=1):
        print(f"\n{posicao}. {item['token']} | {item['similaridade']:.4f}")
        print(f"   {item['frase']}")
    print(f"\nDica:\n{obter_dica(resultado['token'])}")
    print("----------------------------------------")


def executar_teste_manual(classificador: PuzzleClassifier) -> None:
    """Executa o conjunto manual e imprime uma métrica exploratória."""
    try:
        dataset = pd.read_csv(MANUAL_TEST_PATH)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Dataset manual não encontrado: {MANUAL_TEST_PATH}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("O dataset manual está vazio.") from exc

    colunas = {"frase", "token_esperado"}
    if not colunas.issubset(dataset.columns):
        faltantes = sorted(colunas - set(dataset.columns))
        raise ValueError(f"Colunas obrigatórias ausentes no dataset manual: {faltantes}")
    if dataset.empty:
        raise ValueError("O dataset manual não possui registros.")

    erros = []
    acertos = 0
    for _, registro in dataset.iterrows():
        pergunta = str(registro["frase"])
        token_esperado = str(registro["token_esperado"])
        resultado = classificador.classificar(pergunta)
        if resultado["token"] == token_esperado:
            acertos += 1
        else:
            erros.append(
                {
                    "pergunta": pergunta,
                    "token_esperado": token_esperado,
                    "token_previsto": resultado["token"],
                    "similaridade": resultado["similaridade"],
                }
            )

    total = len(dataset)
    acuracia = acertos / total if total else 0.0
    print("\n========================================")
    print("        TESTE MANUAL EXPLORATÓRIO")
    print("========================================")
    print(f"Total de perguntas: {total}")
    print(f"Acertos: {acertos}")
    print(f"Erros: {len(erros)}")
    print(f"Acurácia preliminar: {acuracia:.2%}")
    print("\nErros:")
    if not erros:
        print("Nenhum erro encontrado.")
    else:
        for erro in erros:
            print(f"\nPergunta: {erro['pergunta']}")
            print(f"Token esperado: {erro['token_esperado']}")
            print(f"Token previsto: {erro['token_previsto']}")
            print(f"Similaridade: {erro['similaridade']:.4f}")


def executar_interativo(classificador: PuzzleClassifier) -> None:
    """Executa o modo interativo até o comando sair ou EOF."""
    print("========================================")
    print("     SISTEMA DE DICAS - PROTÓTIPO")
    print("========================================")
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


def main() -> int:
    """Executa o modo interativo ou o teste manual conforme o argumento."""
    parser = argparse.ArgumentParser(description="Protótipo de dicas por embeddings.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Executa as perguntas de data/teste_manual.csv.",
    )
    args = parser.parse_args()

    try:
        classificador = PuzzleClassifier(DATASET_PATH)
        if args.test:
            executar_teste_manual(classificador)
        else:
            executar_interativo(classificador)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Erro ao iniciar o protótipo: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
