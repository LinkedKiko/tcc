"""Configurações centrais do protótipo."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATASET_PATH = PROJECT_ROOT / "data" / "frases_referencia.csv"
MANUAL_TEST_PATH = PROJECT_ROOT / "data" / "teste_manual.csv"
TOP_K = 3

# O limiar será escolhido posteriormente por experimentos com dados de avaliação.
# Enquanto for None, a decisão usa normalmente a maior similaridade encontrada.
SIMILARITY_THRESHOLD = None
