"""Carregamento, validação e valores derivados da configuração do puzzle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.config import PUZZLE_CONFIG_PATH


class PuzzleConfigError(ValueError):
    """Erro de validação da configuração estática do puzzle."""


@dataclass(frozen=True)
class AnimalCount:
    """Quantidades de um animal nas partes visível e fotografada do mural."""

    visivel: int
    fotografia: int

    @property
    def total(self) -> int:
        """Retorna a quantidade calculada a partir das duas partes."""
        return self.visivel + self.fotografia


@dataclass(frozen=True)
class PuzzleConfig:
    """Configuração imutável e validada da sala dos símbolos.

    As senhas não são armazenadas no JSON. ``senha_cofre`` e ``senha_porta``
    são propriedades calculadas a partir do horário e das pistas do puzzle.
    """

    nome: str
    horario: str
    animais: Mapping[str, AnimalCount]
    enciclopedia: Mapping[str, str]
    sequencia: tuple[str, ...]

    @classmethod
    def from_file(cls, path: Path | str = PUZZLE_CONFIG_PATH) -> "PuzzleConfig":
        """Carrega e valida um arquivo JSON, com erros explicativos."""
        caminho = Path(path)
        try:
            with caminho.open("r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)
        except FileNotFoundError as exc:
            raise PuzzleConfigError(
                f"Arquivo de configuração do puzzle não encontrado: {caminho}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PuzzleConfigError(
                f"JSON inválido na configuração do puzzle ({caminho}): {exc.msg}."
            ) from exc
        except OSError as exc:
            raise PuzzleConfigError(
                f"Não foi possível ler a configuração do puzzle: {exc}"
            ) from exc
        return cls.from_dict(dados)

    @classmethod
    def from_dict(cls, dados: Mapping[str, Any]) -> "PuzzleConfig":
        """Valida um dicionário e cria uma configuração pronta para uso."""
        if not isinstance(dados, Mapping):
            raise PuzzleConfigError("A configuração do puzzle deve ser um objeto JSON.")

        nome = _texto_obrigatorio(dados, "nome")
        relogio = _objeto_obrigatorio(dados, "relogio")
        horario = _validar_horario(_texto_obrigatorio(relogio, "horario"))

        mural = _objeto_obrigatorio(dados, "mural")
        animais_dados = _objeto_obrigatorio(mural, "animais")
        animais: dict[str, AnimalCount] = {}
        for animal, valores in animais_dados.items():
            if not isinstance(animal, str) or not animal.strip():
                raise PuzzleConfigError("Cada animal do mural precisa ter um identificador.")
            objeto = _objeto_obrigatorio(animais_dados, animal)
            visivel = _quantidade(objeto, "visivel", animal)
            fotografia = _quantidade(objeto, "fotografia", animal)
            animais[animal] = AnimalCount(visivel, fotografia)
        if not animais:
            raise PuzzleConfigError("O mural precisa possuir ao menos um animal.")

        enciclopedia = _objeto_obrigatorio(dados, "enciclopedia")
        associacoes: dict[str, str] = {}
        for simbolo, animal in enciclopedia.items():
            if not isinstance(simbolo, str) or not simbolo.strip():
                raise PuzzleConfigError("Cada símbolo da enciclopédia precisa ser texto não vazio.")
            if not isinstance(animal, str) or not animal.strip():
                raise PuzzleConfigError(
                    f"O símbolo {simbolo!r} precisa apontar para um animal válido."
                )
            if animal not in animais:
                raise PuzzleConfigError(
                    f"O animal {animal!r} da enciclopédia não existe no mural."
                )
            associacoes[simbolo] = animal
        if not associacoes:
            raise PuzzleConfigError("A enciclopédia precisa possuir ao menos um símbolo.")

        parede = _objeto_obrigatorio(dados, "parede")
        sequencia_dados = parede.get("sequencia")
        if not isinstance(sequencia_dados, list) or not sequencia_dados:
            raise PuzzleConfigError("parede.sequencia deve ser uma lista não vazia.")
        sequencia: list[str] = []
        for simbolo in sequencia_dados:
            if not isinstance(simbolo, str) or not simbolo.strip():
                raise PuzzleConfigError("Todos os símbolos da parede devem ser textos não vazios.")
            if simbolo not in associacoes:
                raise PuzzleConfigError(
                    f"O símbolo {simbolo!r} da parede não existe na enciclopédia."
                )
            sequencia.append(simbolo)

        return cls(
            nome=nome,
            horario=horario,
            animais=animais,
            enciclopedia=associacoes,
            sequencia=tuple(sequencia),
        )

    @property
    def combinacao_cofre(self) -> str:
        """Alias semântico para a combinação numérica do cofre."""
        return self.senha_cofre

    @property
    def senha_cofre(self) -> str:
        """Combinação do cofre obtida diretamente do horário do relógio."""
        return self.horario.replace(":", "")

    @property
    def totais_animais(self) -> dict[str, int]:
        """Quantidades totais, sempre calculadas como visível + fotografia."""
        return {animal: valores.total for animal, valores in self.animais.items()}

    @property
    def senha_porta(self) -> str:
        """Sequência numérica derivada dos símbolos da parede e do mural."""
        totais = self.totais_animais
        return "".join(str(totais[self.enciclopedia[simbolo]]) for simbolo in self.sequencia)

    @property
    def mural(self) -> dict[str, dict[str, dict[str, int]]]:
        """Representação somente leitura prática dos dados do mural."""
        return {
            "animais": {
                animal: {
                    "visivel": valores.visivel,
                    "fotografia": valores.fotografia,
                    "total": valores.total,
                }
                for animal, valores in self.animais.items()
            }
        }


def load_puzzle_config(path: Path | str = PUZZLE_CONFIG_PATH) -> PuzzleConfig:
    """Atalho público para carregar a configuração padrão ou informada."""
    return PuzzleConfig.from_file(path)


def _objeto_obrigatorio(objeto: Mapping[str, Any], chave: str) -> Mapping[str, Any]:
    if chave not in objeto:
        raise PuzzleConfigError(f"Campo obrigatório ausente na configuração: {chave}.")
    valor = objeto[chave]
    if not isinstance(valor, Mapping):
        raise PuzzleConfigError(f"O campo {chave!r} deve ser um objeto JSON.")
    return valor


def _texto_obrigatorio(objeto: Mapping[str, Any], chave: str) -> str:
    if chave not in objeto:
        raise PuzzleConfigError(f"Campo obrigatório ausente na configuração: {chave}.")
    valor = objeto[chave]
    if not isinstance(valor, str) or not valor.strip():
        raise PuzzleConfigError(f"O campo {chave!r} deve ser um texto não vazio.")
    return valor.strip()


def _quantidade(objeto: Mapping[str, Any], chave: str, animal: str) -> int:
    if chave not in objeto:
        raise PuzzleConfigError(f"O animal {animal!r} não possui o campo {chave!r}.")
    valor = objeto[chave]
    if isinstance(valor, bool) or not isinstance(valor, int) or valor < 0:
        raise PuzzleConfigError(
            f"A quantidade {chave!r} do animal {animal!r} deve ser um inteiro não negativo."
        )
    return valor


def _validar_horario(horario: str) -> str:
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", horario) is None:
        raise PuzzleConfigError("O horário deve usar o formato HH:MM:SS.")
    horas, minutos, segundos = (int(parte) for parte in horario.split(":"))
    if horas > 23 or minutos > 59 or segundos > 59:
        raise PuzzleConfigError("O horário possui horas, minutos ou segundos inválidos.")
    return horario
