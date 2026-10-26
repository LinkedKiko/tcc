"""Finaliza o manifesto de uma execução de evaluate já concluída."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.executar_reproducao_semana5 import (
    MODEL_NAME,
    REQUIRED_FILES,
    _hash_tree,
    _report,
    _version,
    sha256,
)


def finalizar(output: Path | None = None) -> dict[str, object]:
    output = ROOT / "results" / "semana5" / "reproducao_final" if output is None else Path(output)
    status_path = output / "status_execucao.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    generated = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name not in {"manifesto_execucao.json", "relatorio_reprodutibilidade.md"}
    }
    missing = [name for name in REQUIRED_FILES if not (output / name).exists()]
    historical_path = ROOT / "results" / "comparacao_metricas.csv"
    current_path = output / "comparacao_metricas.csv"
    comparison: list[dict[str, object]] = []
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
    status_final = "reprodução incompleta"
    if int(status["codigo_saida"]) == 0 and not missing:
        status_final = "reprodução concluída" if all(item["status"] == "coincidente" for item in comparison) else "reprodução concluída com divergências"
    id_execucao = "reproducao_" + datetime.fromisoformat(status["inicio"]).strftime("%Y%m%d_%H%M%S")
    manifest = {
        "id_execucao": id_execucao,
        "status": status_final,
        "codigo_saida": int(status["codigo_saida"]),
        "comando": [
            "PowerShell",
            "$env:HF_HUB_OFFLINE=1",
            "$env:TRANSFORMERS_OFFLINE=1",
            str(Path(sys.executable)), "evaluate.py", "--modo", "completo",
            "--resultados", "results/semana5/reproducao_final", "--sem-graficos",
        ],
        "diretorio_trabalho": str(ROOT),
        "python_executavel": sys.executable,
        "python": platform.python_version(),
        "versoes": {package: _version(package) for package in ("sentence-transformers", "torch", "numpy", "pandas", "scikit-learn", "pytest", "streamlit")},
        "modelo": MODEL_NAME,
        "inicio_utc": status["inicio"],
        "fim_utc": status["fim"],
        "duracao_segundos": status["duracao_segundos"],
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(),
        "hashes_datasets": {
            "data/frases_referencia.csv": sha256(ROOT / "data" / "frases_referencia.csv"),
            "data/teste_manual.csv": sha256(ROOT / "data" / "teste_manual.csv"),
        },
        "artefatos_historicos_nao_confirmados": _hash_tree(ROOT / "results" / "semana5" / "reexecucao_semana2"),
        "arquivos_gerados": generated,
        "arquivos_obrigatorios_ausentes": missing,
        "comparacao_historica": comparison,
        "observacao": "Execução direta registrada por status_execucao.json; tentativas anteriores do wrapper permanecem arquivadas.",
    }
    (output / "manifesto_execucao.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (ROOT / "results" / "semana5" / "relatorio_reprodutibilidade.md").write_text(_report(manifest, output), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = finalizar()
    print(json.dumps({"status": result["status"], "codigo_saida": result["codigo_saida"]}, ensure_ascii=False))
    raise SystemExit(0 if result["status"] != "reprodução incompleta" else 1)
