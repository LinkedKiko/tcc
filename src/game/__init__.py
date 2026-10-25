"""Regras e serviços independentes da interface do jogo."""

from .hint_manager import HintManager, classificar_e_obter_dica
from .puzzle_config import AnimalCount, PuzzleConfig, PuzzleConfigError, load_puzzle_config
from .puzzle_engine import PuzzleEngine
from .puzzle_state import PuzzleState

__all__ = [
    "AnimalCount",
    "HintManager",
    "PuzzleConfig",
    "PuzzleConfigError",
    "PuzzleEngine",
    "PuzzleState",
    "classificar_e_obter_dica",
    "load_puzzle_config",
]
