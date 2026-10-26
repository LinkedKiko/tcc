"""Executa a suíte e os grupos obrigatórios, preservando logs e contagens."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]

COMMANDS = {
    "pytest_completo": [sys.executable, "-m", "pytest", "-q", "--junitxml=__OUTPUT__/pytest_completo.xml"],
    "pytest_streamlit": [sys.executable, "-m", "pytest", "tests/test_streamlit_app.py", "-v", "--junitxml=__OUTPUT__/pytest_streamlit.xml"],
    "pytest_integracao_modelo": [sys.executable, "-m", "pytest", "tests/test_integracao_modelo.py", "-m", "integracao", "-v", "--junitxml=__OUTPUT__/pytest_integracao_modelo.xml"],
    "pytest_semana5": [sys.executable, "-m", "pytest", "tests/test_benchmark_latencia.py", "tests/test_consolidacao_semana5.py", "tests/test_integridade_semana5.py", "-v", "--junitxml=__OUTPUT__/pytest_semana5.xml"],
}


def executar(output: Path | None = None) -> dict[str, object]:
    output = ROOT / "results" / "semana5" / "testes_finais" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    historico = output / "historico_execucoes"
    historico.mkdir(exist_ok=True)
    carimbo = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for path in output.iterdir():
        if path.is_file() and path.name not in {"resumo_testes.json", "relatorio_testes.md"}:
            shutil.copy2(path, historico / f"{carimbo}_{path.name}")
    resultados = {}
    for nome, command in COMMANDS.items():
        command = [argument.replace("__OUTPUT__", str(output)) for argument in command]
        if nome == "pytest_completo":
            command[-1] = f"--junitxml={output / 'pytest_completo.xml'}"
        inicio = datetime.now(timezone.utc)
        clock = time.perf_counter()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        duracao = time.perf_counter() - clock
        fim = datetime.now(timezone.utc)
        log_path = output / f"{nome}.out"
        log_path.write_text(process.stdout + ("\n--- STDERR ---\n" + process.stderr if process.stderr else ""), encoding="utf-8")
        resultados[nome] = {
            "comando": command,
            "inicio_utc": inicio.isoformat(),
            "fim_utc": fim.isoformat(),
            "duracao_segundos": duracao,
            "codigo_saida": process.returncode,
            "resumo_junit": _junit_summary(output / f"{nome}.xml"),
        }
    resumo = {
        "python": platform.python_version(),
        "executavel": sys.executable,
        "diretorio_trabalho": str(ROOT),
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "execucoes": resultados,
    }
    (output / "resumo_testes.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "relatorio_testes.md").write_text(_report(resumo), encoding="utf-8")
    return resumo


def _junit_summary(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    root = ElementTree.parse(path).getroot()
    nodes = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    summary = {key: 0 for key in ("tests", "passed", "failures", "skipped", "errors")}
    for node in nodes:
        summary["tests"] += int(node.attrib.get("tests", 0))
        summary["failures"] += int(node.attrib.get("failures", 0))
        summary["skipped"] += int(node.attrib.get("skipped", 0))
        summary["errors"] += int(node.attrib.get("errors", 0))
    summary["passed"] = summary["tests"] - summary["failures"] - summary["skipped"] - summary["errors"]
    return summary


def _report(resumo: dict[str, object]) -> str:
    lines = [
        "# Relatório dos testes finais — Semana 5",
        "",
        "Cada comando abaixo é uma execução independente; os números não são somados entre execuções. A suíte completa é a única fonte da contagem global.",
        "",
        "| Execução | Código | Coletados | Aprovados | Falhas | Ignorados | Erros | Duração (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nome, item in resumo["execucoes"].items():
        junit = item.get("resumo_junit") or {}
        lines.append(
            f"| `{nome}` | {item['codigo_saida']} | {junit.get('tests', 'n/d')} | {junit.get('passed', 'n/d')} | {junit.get('failures', 'n/d')} | {junit.get('skipped', 'n/d')} | {junit.get('errors', 'n/d')} | {item['duracao_segundos']:.3f} |"
        )
    lines.extend([
        "",
        f"Ambiente: Python `{resumo['python']}`, executável `{resumo['executavel']}`.",
        "Os logs completos estão nos arquivos `.out`; a suíte completa possui `pytest_completo.xml`.",
        "Uma execução interrompida ou com erro de coleta não é apresentada como aprovada.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    result = executar()
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if all(item["codigo_saida"] == 0 for item in result["execucoes"].values()) else 1)
