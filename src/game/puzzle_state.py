"""Estado de runtime de uma partida, sem dependência da interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PuzzleState:
    """Estados mutáveis mínimos necessários para acompanhar a partida."""

    cofre_aberto: bool = False
    foto_encontrada: bool = False
    mural_completo: bool = False
    porta_aberta: bool = False

    @property
    def venceu(self) -> bool:
        """Indica se a porta foi aberta e a partida foi concluída."""
        return self.porta_aberta

    def reiniciar(self) -> None:
        """Restaura o estado inicial sem substituir a instância."""
        self.cofre_aberto = False
        self.foto_encontrada = False
        self.mural_completo = False
        self.porta_aberta = False

    def copiar(self) -> "PuzzleState":
        """Retorna uma cópia para consultas que não exponham o estado interno."""
        return PuzzleState(
            cofre_aberto=self.cofre_aberto,
            foto_encontrada=self.foto_encontrada,
            mural_completo=self.mural_completo,
            porta_aberta=self.porta_aberta,
        )

    def to_dict(self) -> dict[str, bool]:
        """Serializa o estado atual para interfaces e resultados de operações."""
        return {
            "cofre_aberto": self.cofre_aberto,
            "foto_encontrada": self.foto_encontrada,
            "mural_completo": self.mural_completo,
            "porta_aberta": self.porta_aberta,
            "venceu": self.venceu,
        }

    def __getitem__(self, chave: str) -> bool:
        """Permite consultar o snapshot tanto por atributo quanto por chave."""
        valores: dict[str, Any] = self.to_dict()
        try:
            return valores[chave]
        except KeyError as exc:
            raise KeyError(f"Estado desconhece a chave {chave!r}.") from exc

    def get(self, chave: str, padrao: Any = None) -> Any:
        """Versão compatível com consultas de dicionário."""
        return self.to_dict().get(chave, padrao)
