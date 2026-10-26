"""Benchmark reproduzível de latência dos classificadores existentes.

A amostra padrão é estratificada: cinco perguntas de cada token presente no
CSV de avaliação, com semente fixa. Cada execução grava todos os artefatos em
um diretório próprio, identificado por ``id_execucao``; os artefatos históricos
existentes na raiz de ``results/semana5`` não são sobrescritos.

Uso:
    python scripts/benchmark_latencia.py --por-classe 5 --seed 42 \
        --repeticoes 30 --aquecimentos 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier import PuzzleClassifier  # noqa: E402
from src.config import DATASET_PATH, MANUAL_TEST_PATH, MODEL_NAME, MODO_PADRAO, MODO_REJEICAO  # noqa: E402
from src.dataset import carregar_avaliacao, carregar_referencia  # noqa: E402
from src.embeddings import EmbeddingService  # noqa: E402
from src.keyword_classifier import KeywordClassifier  # noqa: E402


EXECUTION_COLUMNS = [
    "id_execucao", "configuracao", "fase", "indice_pergunta", "repeticao",
    "tempo_ms", "threshold", "modelo", "quantidade_perguntas", "aquecimentos",
]
SUMMARY_COLUMNS = [
    "id_execucao", "configuracao", "fase", "n_execucoes", "media_ms",
    "mediana_ms", "desvio_padrao_ms", "minimo_ms", "maximo_ms", "p95_ms",
    "threshold", "status",
]
SAMPLE_COLUMNS = [
    "id_pergunta", "frase", "token_esperado", "posicao_original", "semente",
]


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def criar_id_execucao(agora: datetime | None = None) -> str:
    """Cria um identificador único, legível e gerado no instante da execução."""
    instante = agora or datetime.now(timezone.utc)
    carimbo = instante.strftime("%Y%m%d_%H%M%S_%f")
    return f"benchmark_{carimbo}_{uuid.uuid4().hex[:8]}"


def amostrar_perguntas_estratificadas(
    dataset: pd.DataFrame,
    por_classe: int = 5,
    seed: int = 42,
    classes: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Seleciona uma amostra reprodutível e equilibrada por ``token_esperado``.

    A ordem original do CSV não influencia a seleção. O índice original é
    preservado para auditoria e cada linha recebe um identificador estável.
    """
    if por_classe <= 0:
        raise ValueError("por_classe deve ser um inteiro positivo.")
    obrigatorias = {"frase", "token_esperado"}
    faltantes = sorted(obrigatorias - set(dataset.columns))
    if faltantes:
        raise ValueError(f"Dataset de avaliação sem colunas obrigatórias: {faltantes}")
    if dataset.empty:
        raise ValueError("Dataset de avaliação vazio; não há perguntas para amostrar.")

    dados = dataset.reset_index(drop=False).rename(columns={"index": "posicao_original"})
    dados["token_esperado"] = dados["token_esperado"].astype(str)
    dados["frase"] = dados["frase"].astype(str)
    classes_selecionadas = list(classes) if classes is not None else sorted(dados["token_esperado"].unique())
    if not classes_selecionadas:
        raise ValueError("Nenhuma classe foi encontrada na coluna token_esperado.")
    if len(set(classes_selecionadas)) != len(classes_selecionadas):
        raise ValueError("A lista de classes contém duplicatas.")

    rng = np.random.default_rng(seed)
    selecionados: list[int] = []
    for classe in classes_selecionadas:
        candidatos = dados[dados["token_esperado"] == classe].sort_values(
            ["frase", "posicao_original"], kind="mergesort"
        )
        indices = candidatos.index.to_numpy()
        if len(indices) < por_classe:
            raise ValueError(
                f"A classe {classe!r} possui {len(indices)} exemplos; "
                f"são necessários {por_classe}."
            )
        escolhidos = rng.choice(indices, size=por_classe, replace=False).tolist()
        selecionados.extend(int(indice) for indice in escolhidos)

    # Mistura as classes sem perder a rastreabilidade da posição original.
    rng.shuffle(selecionados)
    amostra = dados.loc[selecionados, ["frase", "token_esperado", "posicao_original"]].copy()
    amostra.insert(0, "id_pergunta", [f"linha_{int(posicao)}" for posicao in amostra["posicao_original"]])
    amostra["semente"] = int(seed)
    amostra = amostra.reset_index(drop=True)
    if amostra["id_pergunta"].duplicated().any():
        raise RuntimeError("A amostragem produziu identificadores duplicados.")
    return amostra[SAMPLE_COLUMNS]


def ambiente_base(
    id_execucao: str,
    repeticoes: int,
    aquecimentos: int,
    amostra: pd.DataFrame,
    seed: int,
    por_classe: int,
) -> dict[str, Any]:
    ambiente: dict[str, Any] = {
        "status": "inicializado",
        "id_execucao": id_execucao,
        "data_execucao_utc": datetime.now(timezone.utc).isoformat(),
        "sistema_operacional": platform.platform(),
        "python": platform.python_version(),
        "processador": platform.processor() or None,
        "cpu_logicos": os.cpu_count(),
        "memoria_gb": None,
        "dispositivo": "desconhecido",
        "sentence_transformers": version("sentence-transformers"),
        "torch": version("torch"),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "matplotlib": version("matplotlib"),
        "streamlit": version("streamlit"),
        "pytest": version("pytest"),
        "modelo": MODEL_NAME,
        "quantidade_frases_referencia": None,
        "quantidade_perguntas_medidas": int(len(amostra)),
        "classes_amostradas": sorted(amostra["token_esperado"].unique().tolist()),
        "contagem_por_classe": {str(k): int(v) for k, v in amostra["token_esperado"].value_counts().sort_index().items()},
        "por_classe": int(por_classe),
        "seed": int(seed),
        "repeticoes": int(repeticoes),
        "aquecimentos": int(aquecimentos),
        "fonte_perguntas": "data/teste_manual.csv, amostragem estratificada por token_esperado",
        "mensagem_erro": None,
    }
    try:
        import psutil  # type: ignore
        ambiente["memoria_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 3)
    except ImportError:
        ambiente["memoria_gb"] = None
    try:
        import torch
        ambiente["dispositivo"] = "cuda" if torch.cuda.is_available() else "cpu"
        ambiente["cuda_disponivel"] = bool(torch.cuda.is_available())
    except Exception:
        ambiente["cuda_disponivel"] = None
    return ambiente


def percentil95(valores: list[float]) -> float | None:
    if not valores:
        return None
    return float(np.percentile(valores, 95)) if len(valores) >= 2 else None


def markdown_table(dataframe: pd.DataFrame) -> str:
    """Renderiza uma tabela Markdown sem depender do pacote opcional tabulate."""
    if dataframe.empty:
        return "(sem registros)"
    colunas = [str(coluna) for coluna in dataframe.columns]
    linhas = [
        "| " + " | ".join(colunas) + " |",
        "| " + " | ".join("---" for _ in colunas) + " |",
    ]
    for valores in dataframe.itertuples(index=False, name=None):
        linhas.append("| " + " | ".join("" if pd.isna(valor) else str(valor) for valor in valores) + " |")
    return "\n".join(linhas)


def resumo_tempos(registros: list[dict[str, Any]], status: str = "executado") -> pd.DataFrame:
    if not registros:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    dataframe = pd.DataFrame(registros)
    linhas = []
    for (id_execucao, configuracao, fase), grupo in dataframe.groupby(
        ["id_execucao", "configuracao", "fase"], sort=False
    ):
        valores = [float(valor) for valor in grupo["tempo_ms"]]
        threshold = grupo["threshold"].iloc[0]
        linhas.append({
            "id_execucao": id_execucao,
            "configuracao": configuracao,
            "fase": fase,
            "n_execucoes": len(valores),
            "media_ms": statistics.mean(valores),
            "mediana_ms": statistics.median(valores),
            "desvio_padrao_ms": statistics.stdev(valores) if len(valores) >= 2 else 0.0,
            "minimo_ms": min(valores),
            "maximo_ms": max(valores),
            "p95_ms": percentil95(valores),
            "threshold": threshold,
            "status": status,
        })
    return pd.DataFrame(linhas, columns=SUMMARY_COLUMNS)


def medir(funcao: Callable[[], Any]) -> float:
    inicio = time.perf_counter()
    funcao()
    return (time.perf_counter() - inicio) * 1000.0


def _sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _registro(
    id_execucao: str,
    nome: str,
    fase: str,
    indice: int,
    repeticao: int,
    tempo: float,
    threshold: float | None,
    quantidade: int,
    aquecimentos: int,
) -> dict[str, Any]:
    return {
        "id_execucao": id_execucao,
        "configuracao": nome,
        "fase": fase,
        "indice_pergunta": indice,
        "repeticao": repeticao,
        "tempo_ms": tempo,
        "threshold": threshold,
        "modelo": MODEL_NAME if nome.startswith("embeddings") else None,
        "quantidade_perguntas": quantidade,
        "aquecimentos": aquecimentos,
    }


def gerar_relatorio_latencia(
    ambiente: dict[str, Any],
    resumo: pd.DataFrame,
    amostra: pd.DataFrame,
    arquivos: dict[str, str] | None = None,
) -> str:
    """Gera o relatório exclusivamente dos metadados e do resumo da execução."""
    linhas = [
        "# Relatório de latência — Semana 5",
        "",
        f"**Identificador da execução:** `{ambiente['id_execucao']}`.",
        f"**Status:** `{ambiente['status']}`.",
        "",
        "Este relatório foi gerado a partir do `ambiente_experimental.json`, "
        "`latencia_execucoes.csv` e `latencia_resumo.csv` desta mesma execução. "
        "Os tempos usam `time.perf_counter()` e não incluem escrita em disco.",
        "",
        "## Amostra estratificada",
        "",
        f"- Perguntas: **{len(amostra)}**; por classe: **{ambiente['por_classe']}**.",
        f"- Semente: **{ambiente['seed']}**.",
        f"- Classes: `{', '.join(ambiente['classes_amostradas'])}`.",
        f"- Distribuição: `{ambiente['contagem_por_classe']}`.",
        "- A amostra completa está em `amostra_perguntas.csv`; a medição preserva "
        "a mesma ordem para os três classificadores.",
        "",
        "## Ambiente",
        "",
    ]
    for chave in (
        "data_execucao_utc", "sistema_operacional", "python", "processador", "cpu_logicos",
        "memoria_gb", "dispositivo", "sentence_transformers", "torch", "numpy", "pandas",
        "scikit_learn", "matplotlib", "streamlit", "pytest", "modelo",
        "quantidade_frases_referencia", "quantidade_perguntas_medidas", "repeticoes", "aquecimentos",
    ):
        linhas.append(f"- **{chave}:** {ambiente.get(chave)!r}")
    linhas.extend(["", "## Medições fora da inferência", ""])
    for chave, descricao in (
        ("tempo_inicializacao_modelo_ms", "Inicialização do modelo"),
        ("tempo_preparacao_embeddings_referencia_ms", "Preparação das referências"),
    ):
        valor = ambiente.get(chave)
        linhas.append(f"- **{descricao}:** {valor:.6f} ms" if isinstance(valor, (int, float)) else f"- **{descricao}:** não executada")
    if ambiente.get("mensagem_erro"):
        linhas.extend(["", "## Motivo da não execução", "", f"`{ambiente['mensagem_erro']}`", "", "Nenhum tempo foi simulado."])
    else:
        linhas.extend([
            "", "## Resumo estatístico", "",
            markdown_table(resumo),
            "",
            "A primeira inferência permanece separada das execuções subsequentes. "
            "O P95 é omitido quando há apenas uma observação.",
        ])
    if arquivos:
        linhas.extend(["", "## Artefatos da execução", ""])
        linhas.extend(f"- `{nome}`: `{caminho}`" for nome, caminho in sorted(arquivos.items()))
    linhas.append("")
    return "\n".join(linhas)


def _manifesto(
    diretorio: Path,
    ambiente: dict[str, Any],
    amostra: pd.DataFrame,
    comando: list[str],
    erro: Exception | None,
) -> dict[str, Any]:
    artefatos = {}
    for caminho in sorted(diretorio.iterdir()):
        if caminho.is_file() and caminho.name != "manifesto_execucao.json":
            artefatos[caminho.name] = {
                "caminho": caminho.name,
                "bytes": caminho.stat().st_size,
                "sha256": _sha256(caminho),
            }
    return {
        "id_execucao": ambiente["id_execucao"],
        "status": ambiente["status"],
        "comando": comando,
        "diretorio_trabalho": str(Path.cwd()),
        "data_execucao_utc": ambiente["data_execucao_utc"],
        "modelo": ambiente["modelo"],
        "seed": ambiente["seed"],
        "por_classe": ambiente["por_classe"],
        "quantidade_perguntas": len(amostra),
        "repeticoes": ambiente["repeticoes"],
        "aquecimentos": ambiente["aquecimentos"],
        "contagem_por_classe": ambiente["contagem_por_classe"],
        "arquivos": artefatos,
        "erro": None if erro is None else f"{type(erro).__name__}: {erro}",
    }


def validar_consistencia_execucao(diretorio: Path) -> dict[str, Any]:
    """Valida a associação entre amostra, CSVs, JSON, relatório e manifesto."""
    diretorio = Path(diretorio)
    ambiente = json.loads((diretorio / "ambiente_experimental.json").read_text(encoding="utf-8"))
    amostra = pd.read_csv(diretorio / "amostra_perguntas.csv")
    execucoes = pd.read_csv(diretorio / "latencia_execucoes.csv")
    resumo = pd.read_csv(diretorio / "latencia_resumo.csv")
    manifesto = json.loads((diretorio / "manifesto_execucao.json").read_text(encoding="utf-8"))
    relatorio = (diretorio / "relatorio_latencia.md").read_text(encoding="utf-8")
    execution_id = ambiente["id_execucao"]
    problemas: list[str] = []
    if manifesto.get("id_execucao") != execution_id:
        problemas.append("manifesto com identificador diferente do ambiente")
    for nome, dataframe in (("execuções", execucoes), ("resumo", resumo)):
        if "id_execucao" not in dataframe.columns or set(dataframe["id_execucao"].dropna()) != {execution_id}:
            problemas.append(f"{nome} com identificador inconsistente")
    if execution_id not in relatorio:
        problemas.append("relatório sem identificador da execução")
    if len(amostra) != ambiente["quantidade_perguntas_medidas"]:
        problemas.append("quantidade de perguntas inconsistente")
    if not amostra["id_pergunta"].is_unique:
        problemas.append("amostra com perguntas duplicadas")
    if not execucoes.empty:
        if set(execucoes["quantidade_perguntas"]) != {len(amostra)}:
            problemas.append("CSV de medições com quantidade de perguntas inconsistente")
        if set(execucoes["aquecimentos"]) != {ambiente["aquecimentos"]}:
            problemas.append("CSV de medições com aquecimentos inconsistente")
    esperado_subsequente = ambiente["repeticoes"] * len(amostra)
    for _, linha in resumo[resumo["fase"] == "subsequente"].iterrows():
        if int(linha["n_execucoes"]) != esperado_subsequente:
            problemas.append(f"resumo incompleto para {linha['configuracao']}")
    return {
        "id_execucao": execution_id,
        "valido": not problemas,
        "problemas": problemas,
        "quantidade_perguntas": len(amostra),
        "repeticoes": ambiente["repeticoes"],
        "modelo": ambiente["modelo"],
        "seed": ambiente["seed"],
    }


def executar_benchmark(
    root: Path = ROOT,
    saida: Path | None = None,
    repeticoes: int = 30,
    aquecimentos: int = 3,
    por_classe: int = 5,
    seed: int = 42,
    max_perguntas: int | None = None,
) -> dict[str, Any]:
    if repeticoes <= 0 or aquecimentos < 0:
        raise ValueError("repeticoes deve ser positiva e aquecimentos não pode ser negativo.")
    if por_classe <= 0:
        raise ValueError("por_classe deve ser positivo.")
    if max_perguntas is not None:
        if max_perguntas <= 0:
            raise ValueError("max_perguntas deve ser positivo.")
        # Compatibilidade segura: o limite antigo não pode voltar a selecionar
        # apenas o prefixo ordenado do CSV. Ele precisa dividir-se igualmente.
        if max_perguntas % 6 != 0:
            raise ValueError(
                "max_perguntas não pode gerar amostra enviesada; use --por-classe "
                "(ou informe um total múltiplo das seis classes)."
            )
        por_classe = max_perguntas // 6

    root = Path(root)
    base_saida = root / "results" / "semana5" / "benchmark_final" if saida is None else Path(saida)
    id_execucao = criar_id_execucao()
    diretorio = base_saida / f"execucao_{id_execucao}"
    diretorio.mkdir(parents=True, exist_ok=False)

    avaliacao_path = root / "data" / "teste_manual.csv"
    referencia_path = root / "data" / "frases_referencia.csv"
    avaliacao = carregar_avaliacao(avaliacao_path)
    amostra = amostrar_perguntas_estratificadas(avaliacao, por_classe=por_classe, seed=seed)
    amostra.to_csv(diretorio / "amostra_perguntas.csv", index=False, encoding="utf-8")
    perguntas = amostra["frase"].astype(str).tolist()
    refs = carregar_referencia(referencia_path)["frase"].astype(str).tolist()
    ambiente = ambiente_base(id_execucao, repeticoes, aquecimentos, amostra, seed, por_classe)
    ambiente["quantidade_frases_referencia"] = len(refs)
    registros: list[dict[str, Any]] = []
    erro: Exception | None = None

    try:
        inicio = time.perf_counter()
        service = EmbeddingService(MODEL_NAME)
        ambiente["tempo_inicializacao_modelo_ms"] = (time.perf_counter() - inicio) * 1000.0

        # A construção real do PuzzleClassifier prepara as referências e
        # reutiliza o serviço já carregado, sem carregar o MiniLM novamente.
        inicio = time.perf_counter()
        classificador = PuzzleClassifier(referencia_path, embedding_service=service)
        ambiente["tempo_preparacao_embeddings_referencia_ms"] = (time.perf_counter() - inicio) * 1000.0
        keyword = KeywordClassifier()
        configuracoes = [
            ("palavras_chave", lambda pergunta: keyword.classificar(pergunta), None),
            ("embeddings_padrao", lambda pergunta: classificador.classificar(pergunta, modo=MODO_PADRAO), None),
            ("embeddings_rejeicao_0_50", lambda pergunta: classificador.classificar(pergunta, modo=MODO_REJEICAO, threshold=0.50), 0.50),
        ]

        primeira = perguntas[0]
        for nome, funcao, threshold in configuracoes:
            tempo = medir(lambda funcao=funcao: funcao(primeira))
            registros.append(_registro(id_execucao, nome, "primeira_inferencia", 0, 0, tempo, threshold, len(perguntas), aquecimentos))

        for nome, funcao, threshold in configuracoes:
            for _ in range(aquecimentos):
                for pergunta in perguntas:
                    funcao(pergunta)
            for repeticao in range(1, repeticoes + 1):
                for indice, pergunta in enumerate(perguntas):
                    tempo = medir(lambda funcao=funcao, pergunta=pergunta: funcao(pergunta))
                    registros.append(_registro(id_execucao, nome, "subsequente", indice, repeticao, tempo, threshold, len(perguntas), aquecimentos))
        ambiente["status"] = "executado"
    except Exception as exc:  # modelo/dependência/rede podem falhar no ambiente
        erro = exc
        ambiente["status"] = "nao_executado"
        ambiente["mensagem_erro"] = f"{type(exc).__name__}: {exc}"

    execucoes = pd.DataFrame(registros, columns=EXECUTION_COLUMNS)
    execucoes.to_csv(diretorio / "latencia_execucoes.csv", index=False, encoding="utf-8")
    resumo = resumo_tempos(registros, ambiente["status"])
    if resumo.empty and ambiente["status"] != "executado":
        resumo = pd.DataFrame([{
            "id_execucao": id_execucao, "configuracao": "todas", "fase": "nao_executado",
            "n_execucoes": 0, "media_ms": None, "mediana_ms": None,
            "desvio_padrao_ms": None, "minimo_ms": None, "maximo_ms": None,
            "p95_ms": None, "threshold": None, "status": "nao_executado",
        }], columns=SUMMARY_COLUMNS)
    resumo.to_csv(diretorio / "latencia_resumo.csv", index=False, encoding="utf-8")
    (diretorio / "ambiente_experimental.json").write_text(json.dumps(ambiente, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    grafico = diretorio / "grafico_latencia.svg"
    if ambiente["status"] == "executado" and not resumo.empty:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            subsequentes = resumo[resumo["fase"] == "subsequente"]
            figura, eixo = plt.subplots(figsize=(8, 4.5))
            for configuracao, grupo in subsequentes.groupby("configuracao"):
                eixo.bar(configuracao, grupo["mediana_ms"].iloc[0], label=configuracao)
            eixo.set_ylabel("Mediana (ms)")
            eixo.set_title(f"Latência mediana — {id_execucao}")
            eixo.tick_params(axis="x", rotation=20)
            figura.tight_layout()
            figura.savefig(grafico, format="svg")
            plt.close(figura)
        except Exception as exc:
            ambiente["grafico_erro"] = f"{type(exc).__name__}: {exc}"
    arquivos_relatorio = {
        "amostra": "amostra_perguntas.csv",
        "execucoes": "latencia_execucoes.csv",
        "resumo": "latencia_resumo.csv",
        "ambiente": "ambiente_experimental.json",
        "grafico": grafico.name if grafico.exists() else "não gerado",
    }
    (diretorio / "relatorio_latencia.md").write_text(
        gerar_relatorio_latencia(ambiente, resumo, amostra, arquivos_relatorio), encoding="utf-8"
    )
    manifesto = _manifesto(diretorio, ambiente, amostra, [sys.executable, *sys.argv], erro)
    (diretorio / "manifesto_execucao.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "id_execucao": id_execucao,
        "diretorio": diretorio,
        "ambiente": ambiente,
        "amostra": amostra,
        "execucoes": execucoes,
        "resumo": resumo,
        "erro": erro,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=None, help="Diretório-base; cada execução recebe uma subpasta própria.")
    parser.add_argument("--repeticoes", type=int, default=30)
    parser.add_argument("--aquecimentos", type=int, default=3)
    parser.add_argument("--por-classe", type=int, default=5, help="Quantidade de perguntas por token.")
    parser.add_argument("--seed", type=int, default=42, help="Semente da amostragem estratificada.")
    parser.add_argument("--max-perguntas", type=int, default=None, help="Compatibilidade segura: total múltiplo das seis classes; não trunca o CSV.")
    args = parser.parse_args(argv)
    try:
        resultado = executar_benchmark(
            args.root.resolve(), args.saida.resolve() if args.saida else None,
            args.repeticoes, args.aquecimentos, args.por_classe, args.seed, args.max_perguntas,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"Erro ao preparar o benchmark: {exc}", file=sys.stderr)
        return 1
    print(f"Latência: {resultado['ambiente']['status']}")
    print(f"Execução: {resultado['id_execucao']}")
    print(f"Artefatos: {resultado['diretorio']}")
    if resultado["erro"]:
        print(f"Causa: {resultado['ambiente']['mensagem_erro']}")
        return 1
    print(resultado["resumo"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
