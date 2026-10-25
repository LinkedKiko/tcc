"""Configurações centrais do protótipo e dos experimentos da Semana 2."""

from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

DATASET_PATH = DATA_DIR / "frases_referencia.csv"
MANUAL_TEST_PATH = DATA_DIR / "teste_manual.csv"
PUZZLE_CONFIG_PATH = DATA_DIR / "puzzle_config.json"

# ---------------------------------------------------------------------------
# Modelo de embeddings
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Quantidade de frases de referência exibidas no ranking de cada classificação.
TOP_K = 3

# Limiar global do modo padrão. Permanece None para preservar o comportamento
# da Semana 1: a decisão usa simplesmente a maior similaridade encontrada.
# O valor escolhido experimentalmente é registrado em results/ e aplicado
# apenas no modo de rejeição, nunca sobrescrevendo esta constante.
SIMILARITY_THRESHOLD: Optional[float] = None

# ---------------------------------------------------------------------------
# Tokens de decisão
# ---------------------------------------------------------------------------
TOKEN_NAO_RELACIONADO = "TOKEN_NAO_RELACIONADO"

# As cinco intenções efetivamente ligadas aos elementos da sala.
PUZZLE_TOKENS: Tuple[str, ...] = (
    "TOKEN_SIMBOLOS",
    "TOKEN_LIVRO",
    "TOKEN_MURAL",
    "TOKEN_COFRE",
    "TOKEN_RELOGIO",
)

# Ordem fixa usada em matrizes de confusão e relatórios por classe.
ALL_TOKENS: Tuple[str, ...] = PUZZLE_TOKENS + (TOKEN_NAO_RELACIONADO,)

# ---------------------------------------------------------------------------
# Modos de decisão do classificador de embeddings
# ---------------------------------------------------------------------------
# "padrao":  compara a pergunta com todas as frases de referência do CSV,
#            incluindo as de TOKEN_NAO_RELACIONADO (Abordagem A).
# "rejeicao": compara apenas com as referências das cinco intenções do puzzle
#            e devolve TOKEN_NAO_RELACIONADO quando a maior similaridade fica
#            abaixo do limiar (Abordagem B).
MODO_PADRAO = "padrao"
MODO_REJEICAO = "rejeicao"
MODOS_VALIDOS: Tuple[str, ...] = (MODO_PADRAO, MODO_REJEICAO)

# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------
# Semente única usada em qualquer sorteio do projeto, garantindo divisões
# determinísticas e reprodutíveis.
RANDOM_STATE = 42

# Proporções da divisão estratificada de data/teste_manual.csv.
VALIDATION_SIZE = 0.40
TEST_SIZE = 0.60

# Grade exploratória inicial de limiares. Não representa limites teóricos do
# modelo; serve apenas para observar o comportamento da rejeição.
THRESHOLDS: Tuple[float, ...] = (
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
)

# Nomes dos conjuntos usados nos relatórios.
CONJUNTO_VALIDACAO = "validacao"
CONJUNTO_TESTE = "teste"
CONJUNTO_COMPLETO = "completo"
