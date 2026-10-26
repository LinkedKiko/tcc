"""Executa e documenta uma reprodução verificável dos experimentos da Semana 2."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REQUIRED_FILES = (
    "particao_validacao.csv",
    "particao_teste.csv",
    "threshold_selecionado.csv",
    "threshold_selecionado_teste.csv",
    "baseline_teste_metricas.csv",
    "embeddings_teste_metricas.csv",
    "embeddings_threshold_teste_metricas.csv",
    "comparacao_metricas.csv",
    "relatorio_semana2.md",
)


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    packages = ("sentence-transformers", "torch", "numpy", "pandas", "scikit-learn", "pytest", "streamlit")
    return {package: _version(package) for package in packages}


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def run_reproduction(root: Path = ROOT, output: Path | None = None) -> dict[str, object]:
    root = Path(root)
    output = root / "results" / "semana5" / "reproducao_final" if output is None else Path(output)
    if output.exists() and any(output.iterdir()):
        archive = output.with_name(
            output.name + "_historico_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
        shutil.copytree(output, archive)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "evaluate.py", "--modo", "completo", "--resultados", str(output), "--sem-graficos"]
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    process = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    duration = time.perf_counter() - clock
    finished = datetime.now(timezone.utc)
    log = process.stdout + ("\n--- STDERR ---\n" + process.stderr if process.stderr else "")
    (output / "execucao.log").write_text(log, encoding="utf-8")
    (output / "comando.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    generated = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"manifesto_execucao.json", "relatorio_reprodutibilidade.md"}:
            generated[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    missing = [name for name in REQUIRED_FILES if not (output / name).exists()]

    comparison: list[dict[str, object]] = []
    historical_path = root / "results" / "comparacao_metricas.csv"
    current_path = output / "comparacao_metricas.csv"
    if historical_path.exists() and current_path.exists():
        historical = pd.read_csv(historical_path)
        current = pd.read_csv(current_path)
        keys = ["metodo", "acuracia", "macro_f1"]
        for metodo in sorted(set(historical["metodo"]) | set(current["metodo"])):
            old = historical[historical["metodo"] == metodo]
            new = current[current["metodo"] == metodo]
            if old.empty or new.empty:
                comparison.append({"metodo": metodo, "status": "ausente_em_uma_execucao"})
                continue
            equal = all(str(old.iloc[0][key]) == str(new.iloc[0][key]) for key in keys)
            comparison.append({
                "metodo": metodo,
                "status": "coincidente" if equal else "divergente",
                "historico": {key: old.iloc[0][key] for key in keys},
                "reproducao": {key: new.iloc[0][key] for key in keys},
            })

    status = "reprodução incompleta"
    if process.returncode == 0 and not missing:
        status = "reprodução concluída" if all(item["status"] == "coincidente" for item in comparison) else "reprodução concluída com divergências"
    manifest = {
        "status": status,
        "codigo_saida": process.returncode,
        "comando": command,
        "diretorio_trabalho": str(root),
        "python_executavel": sys.executable,
        "python": platform.python_version(),
        "versoes": package_versions(),
        "modelo": MODEL_NAME,
        "inicio_utc": started.isoformat(),
        "fim_utc": finished.isoformat(),
        "duracao_segundos": duration,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False).stdout.strip(),
        "hashes_datasets": {
            "data/frases_referencia.csv": sha256(root / "data" / "frases_referencia.csv"),
            "data/teste_manual.csv": sha256(root / "data" / "teste_manual.csv"),
        },
        "artefatos_historicos_nao_confirmados": _hash_tree(root / "results" / "semana5" / "reexecucao_semana2"),
        "arquivos_gerados": generated,
        "arquivos_obrigatorios_ausentes": missing,
        "comparacao_historica": comparison,
    }
    (output / "manifesto_execucao.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (root / "results" / "semana5" / "relatorio_reprodutibilidade.md").write_text(_report(manifest, output), encoding="utf-8")
    return manifest


def _hash_tree(directory: Path) -> dict[str, object]:
    if not directory.exists():
        return {"presente": False}
    return {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(directory.iterdir()) if path.is_file()}


def _report(manifest: dict[str, object], output: Path) -> str:
    lines = [
        "# Relatório de reprodutibilidade — Semana 5",
        "",
        f"**Status:** **{manifest['status']}**.",
        f"**Identificador da execução:** `{manifest['id_execucao']}`.",
        "",
        "A procedência dos arquivos existentes em `reexecucao_semana2/` foi registrada por hash, mas sua existência isolada não foi tratada como prova de execução concluída.",
        "",
        "## Execução verificável",
        "",
        f"- Comando: `{ ' '.join(manifest['comando']) }`",
        f"- Código de saída: **{manifest['codigo_saida']}**",
        f"- Diretório: `{output}`",
        f"- Commit observado: `{manifest['commit'] or 'não disponível'}`",
        f"- Duração: **{manifest['duracao_segundos']:.3f} s**",
        f"- Modelo: `{manifest['modelo']}`",
        f"- Python: `{manifest['python']}`",
        "- Partições esperadas: validação 40 perguntas, teste 60 perguntas, com seleção do threshold somente na validação.",
        "",
        "## Arquivos",
        "",
        f"- Obrigatórios ausentes: `{manifest['arquivos_obrigatorios_ausentes']}`.",
        f"- Arquivos gerados: **{len(manifest['arquivos_gerados'])}**; hashes no `manifesto_execucao.json`.",
        "",
        "## Comparação com o histórico",
        "",
        "| Método | Situação | Histórico | Reprodução |",
        "|---|---|---|---|",
    ]
    for item in manifest["comparacao_historica"]:
        old = item.get("historico", {})
        new = item.get("reproducao", {})
        lines.append(f"| {item['metodo']} | {item['status']} | {old} | {new} |")
    lines.extend(["", "Nenhum histórico foi sobrescrito. Divergências, se presentes, devem ser investigadas por versões, partições, modelo e implementação.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_reproduction()
    print(json.dumps({"status": result["status"], "codigo_saida": result["codigo_saida"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] != "reprodução incompleta" else 1)
