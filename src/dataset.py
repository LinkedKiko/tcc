"""Carregamento e particionamento do conjunto de avaliação.

O arquivo ``data/frases_referencia.csv`` é a base de comparação do
classificador semântico e nunca é alterado nem usado como conjunto de
avaliação. O arquivo ``data/teste_manual.csv`` contém perguntas independentes
e é dividido de forma determinística em validação (40%) e teste (60%).

A divisão é estratificada por ``token_esperado`` e usa a semente única
``RANDOM_STATE``. Rodar o programa novamente produz exatamente as mesmas
partições, o que torna os experimentos reprodutíveis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ALL_TOKENS,
    CONJUNTO_TESTE,
    CONJUNTO_VALIDACAO,
    DATASET_PATH,
    MANUAL_TEST_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    VALIDATION_SIZE,
)

COLUNAS_AVALIACAO = ("frase", "token_esperado")
DIFICULDADE_PADRAO = "nao_informada"

ARQUIVO_VALIDACAO = "particao_validacao.csv"
ARQUIVO_TESTE = "particao_teste.csv"
ARQUIVO_RESUMO = "particao_resumo.csv"


@dataclass(frozen=True)
class Particoes:
    """Resultado da divisão determinística do conjunto de avaliação."""

    completo: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    random_state: int
    validation_size: float

    def resumo(self) -> pd.DataFrame:
        """Tabela com a contagem de perguntas por token e partição."""
        linhas = []
        for token in ALL_TOKENS:
            linhas.append(
                {
                    "token_esperado": token,
                    "completo": int((self.completo["token_esperado"] == token).sum()),
                    "validacao": int((self.validacao["token_esperado"] == token).sum()),
                    "teste": int((self.teste["token_esperado"] == token).sum()),
                }
            )
        resumo = pd.DataFrame(linhas)
        total = {
            "token_esperado": "TOTAL",
            "completo": int(len(self.completo)),
            "validacao": int(len(self.validacao)),
            "teste": int(len(self.teste)),
        }
        return pd.concat([resumo, pd.DataFrame([total])], ignore_index=True)

    def obter(self, conjunto: str) -> pd.DataFrame:
        """Devolve a partição pelo nome: ``validacao``, ``teste`` ou ``completo``."""
        mapa = {
            CONJUNTO_VALIDACAO: self.validacao,
            CONJUNTO_TESTE: self.teste,
            "completo": self.completo,
        }
        if conjunto not in mapa:
            raise ValueError(
                f"Conjunto inválido: {conjunto!r}. Use um de {sorted(mapa)}."
            )
        return mapa[conjunto]


def carregar_referencia(caminho: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Carrega o CSV de frases de referência apenas para leitura e conferência."""
    dataset = _ler_csv(caminho, "referência")
    obrigatorias = {"frase", "token"}
    faltantes = sorted(obrigatorias - set(dataset.columns))
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV de referência: {faltantes}")
    dataset = dataset.copy()
    dataset["frase"] = dataset["frase"].astype(str).str.strip()
    dataset["token"] = dataset["token"].astype(str).str.strip()
    return dataset


def carregar_avaliacao(caminho: Path | str = MANUAL_TEST_PATH) -> pd.DataFrame:
    """Carrega e valida o conjunto de avaliação rotulado manualmente.

    Quando a coluna ``dificuldade`` não existe, ela é criada com o valor
    ``"nao_informada"`` para manter o formato dos relatórios.
    """
    dataset = _ler_csv(caminho, "de avaliação")
    faltantes = sorted(set(COLUNAS_AVALIACAO) - set(dataset.columns))
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset de avaliação: {faltantes}")

    dataset = dataset.copy()
    if "dificuldade" not in dataset.columns:
        dataset["dificuldade"] = DIFICULDADE_PADRAO

    for coluna in ("frase", "token_esperado", "dificuldade"):
        dataset[coluna] = dataset[coluna].astype(str).str.strip()

    dataset["dificuldade"] = dataset["dificuldade"].replace(
        {"": DIFICULDADE_PADRAO, "nan": DIFICULDADE_PADRAO}
    )

    if (dataset["frase"] == "").any():
        raise ValueError("O dataset de avaliação possui frases vazias.")
    if (dataset["token_esperado"] == "").any():
        raise ValueError("O dataset de avaliação possui tokens esperados vazios.")

    desconhecidos = sorted(set(dataset["token_esperado"]) - set(ALL_TOKENS))
    if desconhecidos:
        raise ValueError(f"Tokens esperados desconhecidos no dataset: {desconhecidos}")

    duplicadas = dataset["frase"].duplicated()
    if duplicadas.any():
        exemplos = dataset.loc[duplicadas, "frase"].head(3).tolist()
        raise ValueError(f"O dataset de avaliação possui frases repetidas: {exemplos}")

    return dataset[["frase", "token_esperado", "dificuldade"]].reset_index(drop=True)


def dividir_validacao_teste(
    dataset: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> Particoes:
    """Divide o conjunto de avaliação de forma estratificada e determinística.

    Args:
        dataset: saída de :func:`carregar_avaliacao`.
        validation_size: fração destinada à validação (o restante vai para teste).
        random_state: semente fixa do sorteio.

    Raises:
        ValueError: se o dataset for pequeno demais para estratificar ou se
            alguma classe tiver menos de duas frases.
    """
    if dataset.empty:
        raise ValueError("Não é possível dividir um dataset vazio.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size precisa estar entre 0 e 1.")

    contagens = dataset["token_esperado"].value_counts()
    insuficientes = sorted(contagens[contagens < 2].index)
    if insuficientes:
        raise ValueError(
            "Cada token precisa de ao menos duas frases para a divisão "
            f"estratificada. Tokens insuficientes: {insuficientes}"
        )

    validacao, teste = train_test_split(
        dataset,
        train_size=validation_size,
        random_state=random_state,
        stratify=dataset["token_esperado"],
        shuffle=True,
    )

    validacao = validacao.sort_index().reset_index(drop=True)
    teste = teste.sort_index().reset_index(drop=True)

    sobreposicao = set(validacao["frase"]) & set(teste["frase"])
    if sobreposicao:
        raise RuntimeError(
            f"A divisão produziu sobreposição entre validação e teste: {sorted(sobreposicao)[:3]}"
        )

    return Particoes(
        completo=dataset.reset_index(drop=True),
        validacao=validacao,
        teste=teste,
        random_state=random_state,
        validation_size=validation_size,
    )


def preparar_particoes(
    caminho: Path | str = MANUAL_TEST_PATH,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> Particoes:
    """Carrega o conjunto de avaliação e devolve as partições determinísticas."""
    return dividir_validacao_teste(
        carregar_avaliacao(caminho),
        validation_size=validation_size,
        random_state=random_state,
    )


def salvar_particoes(
    particoes: Particoes,
    diretorio: Path | str = RESULTS_DIR,
) -> Dict[str, Path]:
    """Registra as partições em ``results/`` sem modificar o arquivo original."""
    destino = Path(diretorio)
    destino.mkdir(parents=True, exist_ok=True)

    caminhos = {
        CONJUNTO_VALIDACAO: destino / ARQUIVO_VALIDACAO,
        CONJUNTO_TESTE: destino / ARQUIVO_TESTE,
        "resumo": destino / ARQUIVO_RESUMO,
    }
    particoes.validacao.assign(particao=CONJUNTO_VALIDACAO).to_csv(
        caminhos[CONJUNTO_VALIDACAO], index=False, encoding="utf-8"
    )
    particoes.teste.assign(particao=CONJUNTO_TESTE).to_csv(
        caminhos[CONJUNTO_TESTE], index=False, encoding="utf-8"
    )
    particoes.resumo().to_csv(caminhos["resumo"], index=False, encoding="utf-8")
    return caminhos


def verificar_integridade(
    particoes: Particoes,
    referencia: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Confere as condições de integridade exigidas pelos critérios de aceitação.

    Returns:
        Dicionário com os totais verificados, a lista de frases em comum com o
        dataset de referência e a lista de frases presentes nas duas partições.
    """
    referencia = carregar_referencia() if referencia is None else referencia
    frases_referencia = set(referencia["frase"].astype(str).str.strip())
    frases_avaliacao = set(particoes.completo["frase"])

    return {
        "total_referencia": int(len(referencia)),
        "total_avaliacao": int(len(particoes.completo)),
        "total_validacao": int(len(particoes.validacao)),
        "total_teste": int(len(particoes.teste)),
        "frases_em_comum": sorted(frases_referencia & frases_avaliacao),
        "sobreposicao_particoes": sorted(
            set(particoes.validacao["frase"]) & set(particoes.teste["frase"])
        ),
    }


def _ler_csv(caminho: Path | str, descricao: str) -> pd.DataFrame:
    """Lê um CSV convertendo falhas de leitura em mensagens claras."""
    caminho = Path(caminho)
    try:
        dataset = pd.read_csv(caminho)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset {descricao} não encontrado: {caminho}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"O dataset {descricao} está vazio.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Não foi possível interpretar o dataset {descricao}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"O dataset {descricao} não está em UTF-8: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Não foi possível ler o dataset {descricao}: {exc}") from exc

    if dataset.empty:
        raise ValueError(f"O dataset {descricao} não possui registros.")
    return dataset


def contar_por_token(dataset: pd.DataFrame, coluna: str = "token_esperado") -> pd.Series:
    """Contagem de frases por token na ordem fixa de ``ALL_TOKENS``."""
    contagem = dataset[coluna].value_counts()
    return pd.Series(
        {token: int(contagem.get(token, 0)) for token in ALL_TOKENS},
        name="quantidade",
    )


def distribuicao_dificuldade(dataset: pd.DataFrame) -> pd.DataFrame:
    """Tabela cruzada entre token esperado e nível de dificuldade."""
    return pd.crosstab(dataset["token_esperado"], dataset["dificuldade"])


__all__ = [
    "Particoes",
    "carregar_avaliacao",
    "carregar_referencia",
    "contar_por_token",
    "distribuicao_dificuldade",
    "dividir_validacao_teste",
    "preparar_particoes",
    "salvar_particoes",
    "verificar_integridade",
]
