"""Demonstração de terminal da lógica da Sala dos Símbolos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from src.classifier import PuzzleClassifier
from src.config import DATASET_PATH, MODO_PADRAO, MODO_REJEICAO, PUZZLE_CONFIG_PATH, TOP_K
from src.game import HintManager, PuzzleConfig, PuzzleEngine


MENU = """\n========================================
           SALA DOS SIMBOLOS
========================================

Voce esta em uma sala trancada.

Existe uma porta com um painel numerico.

Ao seu redor, voce observa:

1. Relogio
2. Cofre
3. Mural
4. Enciclopedia
5. Simbolos na parede
6. Porta

7. Fazer uma pergunta
8. Consultar estado da partida
9. Reiniciar jogo
0. Sair
"""


class TerminalGame:
    """Adaptador de entrada/saída para o PuzzleEngine."""

    def __init__(self, engine: PuzzleEngine, classificador: Any) -> None:
        self.engine = engine
        self.classificador = classificador

    def executar(self) -> None:
        print(MENU)
        while True:
            try:
                opcao = input("Escolha uma opcao: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando o jogo.")
                return

            if opcao == "0":
                print("Encerrando o jogo.")
                return
            try:
                continuar = self._executar_opcao(opcao)
            except (EOFError, KeyboardInterrupt):
                print("\nEncerrando o jogo.")
                return
            except (ValueError, TypeError, RuntimeError) as exc:
                print(f"Erro: {exc}")
                continuar = True
            if not continuar:
                return

    def _executar_opcao(self, opcao: str) -> bool:
        if opcao in {"1", "3", "4", "5"}:
            objeto = {"1": "relogio", "3": "mural", "4": "enciclopedia", "5": "simbolos"}[opcao]
            self._mostrar(self.engine.examinar_objeto(objeto))
            if opcao == "3" and self.engine.obter_estado().get("foto_encontrada"):
                if self._confirmar("Deseja completar o mural agora? (s/n): "):
                    self._mostrar(self.engine.completar_mural())
            return True
        if opcao == "2":
            self._mostrar(self.engine.examinar_objeto("cofre"))
            estado = self.engine.obter_estado()
            if not estado.cofre_aberto and self._confirmar("Deseja inserir uma combinacao? (s/n): "):
                self._mostrar(self.engine.abrir_cofre(input("Combinacao do cofre: ")))
            if self.engine.obter_estado().cofre_aberto and not self.engine.obter_estado().foto_encontrada:
                if self._confirmar("Deseja retirar a fotografia? (s/n): "):
                    self._mostrar(self.engine.encontrar_fotografia())
            return True
        if opcao == "6":
            self._mostrar(self.engine.examinar_objeto("porta"))
            if not self.engine.obter_estado().porta_aberta and self._confirmar("Deseja inserir uma senha? (s/n): "):
                resultado = self.engine.abrir_porta(input("Senha da porta: "))
                self._mostrar(resultado)
                if resultado["evento"] == "VITORIA":
                    print("\nParabens! A partida foi concluida.")
            return True
        if opcao == "7":
            pergunta = input("\nDigite sua pergunta:\n> ")
            if not pergunta.strip():
                print("A pergunta nao pode ser vazia.")
                return True
            resultado = self.engine.processar_pergunta(pergunta, self.classificador)
            classificacao = resultado["classificacao"]
            print(f"\nToken identificado:\n{resultado['token']}")
            if classificacao.get("similaridade") is not None:
                print(f"Similaridade: {classificacao['similaridade']:.4f}")
            print(f"\nDica:\n{resultado['dica']}")
            return True
        if opcao == "8":
            self._mostrar_estado()
            return True
        if opcao == "9":
            self._mostrar(self.engine.reiniciar())
            return True
        print("Comando invalido. Escolha uma opcao do menu.")
        return True

    @staticmethod
    def _confirmar(pergunta: str) -> bool:
        resposta = input(pergunta).strip().casefold()
        return resposta in {"s", "sim", "y", "yes"}

    @staticmethod
    def _mostrar(resultado: dict[str, Any]) -> None:
        print(f"\n{resultado['mensagem']}")

    def _mostrar_estado(self) -> None:
        estado = self.engine.obter_estado()
        print("\nEstado da partida:")
        print(f"Cofre: {'Aberto' if estado.cofre_aberto else 'Fechado'}")
        print(f"Fotografia: {'Encontrada' if estado.foto_encontrada else 'Nao encontrada'}")
        print(f"Mural: {'Completo' if estado.mural_completo else 'Incompleto'}")
        print(f"Porta: {'Aberta' if estado.porta_aberta else 'Fechada'}")
        if estado.venceu:
            print("Vitoria: registrada")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jogo de terminal da Sala dos Simbolos.")
    parser.add_argument("--config", type=Path, default=PUZZLE_CONFIG_PATH, help="JSON da configuração do puzzle.")
    parser.add_argument("--referencia", type=Path, default=DATASET_PATH, help="CSV de frases de referência.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Quantidade de referências mantida no ranking.")
    parser.add_argument("--threshold", type=float, default=None, help="Ativa o modo de rejeição com este limiar.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        config = PuzzleConfig.from_file(args.config)
        modo = MODO_REJEICAO if args.threshold is not None else MODO_PADRAO
        classificador = PuzzleClassifier(args.referencia, top_k=args.top_k)
        engine = PuzzleEngine(config, hint_manager=HintManager())
        TerminalGame(engine, _ClassificadorTerminal(classificador, modo, args.threshold)).executar()
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Erro ao iniciar o jogo: {exc}", file=sys.stderr)
        return 1
    return 0


class _ClassificadorTerminal:
    """Fixa modo e limiar sem alterar o classificador da Semana 2."""

    def __init__(self, classificador: PuzzleClassifier, modo: str, threshold: Optional[float]) -> None:
        self.classificador = classificador
        self.modo = modo
        self.threshold = threshold

    def classificar(self, pergunta: str) -> dict[str, Any]:
        return self.classificador.classificar(pergunta, self.modo, self.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
