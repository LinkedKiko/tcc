"""Valida a integridade e a rastreabilidade dos dados da Semana 5.

O script somente lê os datasets e artefatos históricos. A normalização é
aplicada exclusivamente às comparações de integridade; nenhum texto, rótulo ou
arquivo de entrada é corrigido automaticamente.

Uso:
    python scripts/validar_integridade_semana5.py
    python scripts/validar_integridade_semana5.py --saida results/semana5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ALL_TOKENS, RANDOM_STATE, VALIDATION_SIZE  # noqa: E402
from src.dataset import preparar_particoes  # noqa: E402


REFERENCE_EXPECTED = {
    "total": 90,
    "por_token": {token: 15 for token in ALL_TOKENS},
}
EVALUATION_EXPECTED = {
    "total": 100,
    "por_token": {
        "TOKEN_SIMBOLOS": 15,
        "TOKEN_LIVRO": 15,
        "TOKEN_MURAL": 15,
        "TOKEN_COFRE": 15,
        "TOKEN_RELOGIO": 15,
        "TOKEN_NAO_RELACIONADO": 25,
    },
}

REQUIRED_REFERENCE = ("frase", "token")
REQUIRED_EVALUATION = ("frase", "token_esperado")

# Termos usados apenas para apontar sobreposições candidatas à revisão humana.
# Eles não alteram rótulos e não representam uma anotação definitiva.
AMBIGUITY_TERMS = {
    "TOKEN_SIMBOLOS": (
        "simbolo", "símbolo", "parede", "porta", "entrada", "saida", "saída",
        "fechadura", "painel", "sequencia", "sequência", "ordem", "marca", "sinal",
        "desenho", "rabisco", "inscricao", "inscrição", "anotacao", "anotação",
    ),
    "TOKEN_COFRE": (
        "cofre", "caixa", "cadeado", "compartimento", "conteudo", "conteúdo",
        "senha", "codigo", "código", "combinacao", "combinação", "destrancar",
        "abrir", "objeto lacrado",
    ),
    "TOKEN_MURAL": (
        "mural", "animal", "animais", "bicho", "bichos", "criatura", "criaturas",
        "figura", "figuras", "quantidade", "contar", "contagem", "imagem", "painel",
        "desenho", "arte", "valores",
    ),
    "TOKEN_LIVRO": (
        "livro", "enciclopedia", "enciclopédia", "pagina", "página", "paginas", "páginas",
        "traducao", "tradução", "correspondencia", "correspondência", "dicionario", "dicionário",
        "leitura", "volume",
    ),
    "TOKEN_RELOGIO": (
        "relogio", "relógio", "ponteiro", "ponteiros", "horario", "horário", "hora",
        "horas", "minuto", "minutos", "segundo", "segundos", "tempo", "instante",
    ),
}
TARGET_AMBIGUOUS_PAIRS = {
    frozenset(pair)
    for pair in (
        ("TOKEN_RELOGIO", "TOKEN_COFRE"),
        ("TOKEN_COFRE", "TOKEN_MURAL"),
        ("TOKEN_MURAL", "TOKEN_LIVRO"),
        ("TOKEN_LIVRO", "TOKEN_SIMBOLOS"),
        ("TOKEN_SIMBOLOS", "TOKEN_COFRE"),
    )
}


def normalizar_frase(valor: Any) -> str:
    """Normaliza somente para comparação: Unicode, acentos, caixa e espaços."""
    texto = "" if valor is None else str(valor)
    texto = unicodedata.normalize("NFKC", texto).casefold()
    decomposed = unicodedata.normalize("NFKD", texto)
    texto = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(texto.split())


def sha256(caminho: Path) -> str | None:
    if not caminho.exists():
        return None
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return digest.hexdigest()


def verificar_csv(
    caminho: Path,
    colunas: tuple[str, ...],
    descricao: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    problemas: list[str] = []
    if not caminho.exists():
        return None, [f"Arquivo ausente: {caminho}"]
    try:
        dataframe = pd.read_csv(caminho)
    except Exception as exc:  # pragma: no cover - mensagem depende do parser
        return None, [f"Falha ao ler {descricao}: {exc}"]
    faltantes = [coluna for coluna in colunas if coluna not in dataframe.columns]
    if faltantes:
        problemas.append(f"Colunas ausentes em {descricao}: {faltantes}")
    if dataframe.empty:
        problemas.append(f"{descricao} não possui registros.")
    for coluna in colunas:
        if coluna in dataframe:
            if dataframe[coluna].isna().any():
                problemas.append(f"Valores ausentes em {descricao}.{coluna}.")
            if (dataframe[coluna].astype(str).str.strip() == "").any():
                problemas.append(f"Valores vazios em {descricao}.{coluna}.")
    return dataframe, problemas


def duplicatas_normalizadas(dataframe: pd.DataFrame, coluna: str) -> list[str]:
    normalizadas = dataframe[coluna].map(normalizar_frase)
    return sorted(normalizadas[normalizadas.duplicated(keep=False)].unique().tolist())


def candidatos_ambiguos(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    candidatos: list[dict[str, Any]] = []
    for linha in dataframe.itertuples(index=False):
        frase = str(linha.frase)
        texto = normalizar_frase(frase)
        encontrados = tuple(
            token
            for token, termos in AMBIGUITY_TERMS.items()
            if any(normalizar_frase(termo) in texto for termo in termos)
        )
        pares = [
            tuple(sorted((encontrado, outro)))
            for encontrado in encontrados
            for outro in encontrados
            if encontrado < outro and frozenset((encontrado, outro)) in TARGET_AMBIGUOUS_PAIRS
        ]
        if pares:
            candidatos.append(
                {
                    "frase": frase,
                    "token_esperado": str(linha.token_esperado),
                    "tokens_sugeridos_por_termos": list(encontrados),
                    "pares_prioritarios": [list(par) for par in sorted(set(pares))],
                    "revisao_humana": True,
                }
            )
    return candidatos


def lista_frases(df: pd.DataFrame, coluna: str) -> set[str]:
    return set(df[coluna].map(normalizar_frase))


def comparar_particao_gerada(
    caminho: Path,
    esperado: pd.DataFrame,
    coluna_token: str = "token_esperado",
) -> dict[str, Any]:
    resultado: dict[str, Any] = {"arquivo": str(caminho), "presente": caminho.exists()}
    if not caminho.exists():
        resultado["compatível"] = False
        resultado["motivo"] = "arquivo não encontrado"
        return resultado
    try:
        gerado = pd.read_csv(caminho)
        if not {"frase", coluna_token}.issubset(gerado.columns):
            resultado["compatível"] = False
            resultado["motivo"] = "colunas incompatíveis"
            return resultado
        esperado_chaves = {
            (normalizar_frase(frase), str(token))
            for frase, token in zip(esperado["frase"], esperado["token_esperado"])
        }
        gerado_chaves = {
            (normalizar_frase(frase), str(token))
            for frase, token in zip(gerado["frase"], gerado[coluna_token])
        }
        resultado["linhas_esperadas"] = len(esperado_chaves)
        resultado["linhas_geradas"] = len(gerado_chaves)
        resultado["compatível"] = esperado_chaves == gerado_chaves
        if not resultado["compatível"]:
            resultado["somente_esperado"] = sorted(esperado_chaves - gerado_chaves)[:10]
            resultado["somente_gerado"] = sorted(gerado_chaves - esperado_chaves)[:10]
        return resultado
    except Exception as exc:
        resultado["compatível"] = False
        resultado["motivo"] = str(exc)
        return resultado


def executar_validacao(root: Path = ROOT, saida: Path | None = None) -> dict[str, Any]:
    data = root / "data"
    results = root / "results"
    saida = results / "semana5" if saida is None else Path(saida)
    saida.mkdir(parents=True, exist_ok=True)

    referencia_path = data / "frases_referencia.csv"
    avaliacao_path = data / "teste_manual.csv"
    referencia, problemas_ref = verificar_csv(referencia_path, REQUIRED_REFERENCE, "referências")
    avaliacao, problemas_av = verificar_csv(avaliacao_path, REQUIRED_EVALUATION, "avaliação")
    problemas = problemas_ref + problemas_av

    checks: dict[str, Any] = {}
    hashes: dict[str, str | None] = {}
    for caminho in (
        referencia_path,
        avaliacao_path,
        data / "avaliacao_funcional_dicas.csv",
        data / "avaliacao_funcional_dicas_resultados.csv",
        data / "puzzle_config.json",
        results / "particao_validacao.csv",
        results / "particao_teste.csv",
        results / "particao_resumo.csv",
    ):
        hashes[str(caminho.relative_to(root))] = sha256(caminho)

    if referencia is not None:
        referencia = referencia.copy()
        referencia["frase"] = referencia["frase"].astype(str).str.strip()
        referencia["token"] = referencia["token"].astype(str).str.strip()
        contagem = referencia["token"].value_counts().to_dict()
        checks["referencias"] = {
            "total": int(len(referencia)),
            "por_token": {token: int(contagem.get(token, 0)) for token in ALL_TOKENS},
            "rotulos_desconhecidos": sorted(set(referencia["token"]) - set(ALL_TOKENS)),
            "duplicatas_normalizadas": duplicatas_normalizadas(referencia, "frase"),
            "valores_ausentes_essenciais": bool(referencia[list(REQUIRED_REFERENCE)].isna().any().any()),
        }
        if len(referencia) != REFERENCE_EXPECTED["total"]:
            problemas.append("Quantidade de referências diferente de 90.")
        if checks["referencias"]["por_token"] != REFERENCE_EXPECTED["por_token"]:
            problemas.append("Distribuição das referências diferente de 15 por token.")
        if checks["referencias"]["rotulos_desconhecidos"]:
            problemas.append("Há rótulos desconhecidos no dataset de referências.")
        if checks["referencias"]["duplicatas_normalizadas"]:
            problemas.append("Há frases duplicadas nas referências após normalização.")

    particoes_info: dict[str, Any] = {}
    ambiguidades: list[dict[str, Any]] = []
    if avaliacao is not None:
        avaliacao = avaliacao.copy()
        avaliacao["frase"] = avaliacao["frase"].astype(str).str.strip()
        avaliacao["token_esperado"] = avaliacao["token_esperado"].astype(str).str.strip()
        contagem = avaliacao["token_esperado"].value_counts().to_dict()
        checks["avaliacao"] = {
            "total": int(len(avaliacao)),
            "por_token": {token: int(contagem.get(token, 0)) for token in ALL_TOKENS},
            "rotulos_desconhecidos": sorted(set(avaliacao["token_esperado"]) - set(ALL_TOKENS)),
            "duplicatas_normalizadas": duplicatas_normalizadas(avaliacao, "frase"),
            "valores_ausentes_essenciais": bool(avaliacao[list(REQUIRED_EVALUATION)].isna().any().any()),
        }
        if len(avaliacao) != EVALUATION_EXPECTED["total"]:
            problemas.append("Quantidade de perguntas diferente de 100.")
        if checks["avaliacao"]["por_token"] != EVALUATION_EXPECTED["por_token"]:
            problemas.append("Distribuição da avaliação diferente da esperada.")
        if checks["avaliacao"]["rotulos_desconhecidos"]:
            problemas.append("Há rótulos desconhecidos na avaliação.")
        if checks["avaliacao"]["duplicatas_normalizadas"]:
            problemas.append("Há frases duplicadas na avaliação após normalização.")

        if referencia is not None:
            comuns = sorted(lista_frases(referencia, "frase") & lista_frases(avaliacao, "frase"))
            checks["frases_idênticas_referencia_avaliacao"] = comuns
            if comuns:
                problemas.append("Existem frases idênticas entre referências e avaliação.")

        try:
            particoes = preparar_particoes(avaliacao_path, validation_size=VALIDATION_SIZE, random_state=RANDOM_STATE)
            frases_validacao = lista_frases(particoes.validacao, "frase")
            frases_teste = lista_frases(particoes.teste, "frase")
            checks["particoes"] = {
                "random_state": RANDOM_STATE,
                "validation_size": VALIDATION_SIZE,
                "total_validacao": int(len(particoes.validacao)),
                "total_teste": int(len(particoes.teste)),
                "sobreposicao_normalizada": sorted(frases_validacao & frases_teste),
                "resumo": particoes.resumo().to_dict(orient="records"),
                "reproducao_deterministica": True,
            }
            if len(particoes.validacao) != 40 or len(particoes.teste) != 60:
                problemas.append("A divisão não produziu 40 perguntas de validação e 60 de teste.")
            if frases_validacao & frases_teste:
                problemas.append("Há frases sobrepostas entre validação e teste.")
            particoes_info["validacao"] = comparar_particao_gerada(results / "particao_validacao.csv", particoes.validacao)
            particoes_info["teste"] = comparar_particao_gerada(results / "particao_teste.csv", particoes.teste)
            if (results / "particao_resumo.csv").exists():
                resumo_gerado = pd.read_csv(results / "particao_resumo.csv")
                esperado_resumo = particoes.resumo()
                checks["resumo_gerado_compativel"] = set(map(tuple, resumo_gerado.astype(str).to_numpy())) == set(map(tuple, esperado_resumo.astype(str).to_numpy()))
                if not checks["resumo_gerado_compativel"]:
                    problemas.append("O resumo de partições gerado difere da divisão reproduzida.")
            for item in particoes_info.values():
                if item.get("presente") and not item.get("compatível"):
                    problemas.append(f"Arquivo de partição incompatível: {item['arquivo']}")
        except Exception as exc:
            checks["particoes"] = {"reproducao_deterministica": False, "erro": str(exc)}
            problemas.append(f"Falha ao reproduzir a divisão: {exc}")
        ambiguidades = candidatos_ambiguos(avaliacao)

    checks["arquivos_gerados"] = particoes_info
    checks["normalizacao"] = "NFKC + NFKD sem diacríticos + casefold + espaços colapsados; usada apenas em comparações"
    checks["hashes_sha256"] = hashes
    checks["arquivos_entrada"] = {
        "referencia": str(referencia_path.relative_to(root)),
        "avaliacao": str(avaliacao_path.relative_to(root)),
    }
    checks["ambiguidades_candidatas"] = ambiguidades
    checks["ambiguidades_observacao"] = "Candidatos são alertas para revisão humana; nenhum rótulo foi alterado automaticamente."
    checks["problemas"] = sorted(set(problemas))
    checks["integridade_aprovada"] = not problemas

    caminho_json = saida / "integridade_dados.json"
    caminho_json.write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    caminho_md = saida / "relatorio_integridade.md"
    caminho_md.write_text(gerar_relatorio(checks), encoding="utf-8")
    return checks


def gerar_relatorio(resultado: dict[str, Any]) -> str:
    linhas = [
        "# Relatório de integridade dos dados — Semana 5",
        "",
        "A verificação é somente leitura. A normalização foi aplicada exclusivamente às comparações e nenhum dataset foi modificado.",
        "",
        f"**Resultado geral:** {'APROVADO' if resultado['integridade_aprovada'] else 'COM PENDÊNCIAS'}",
        "",
        "## Arquivos e hashes SHA-256",
        "",
        "| Arquivo | SHA-256 |",
        "|---|---|",
    ]
    for caminho, digest in resultado["hashes_sha256"].items():
        linhas.append(f"| `{caminho}` | `{digest or 'ausente'}` |")
    linhas.extend(["", "## Referências semânticas", ""])
    referencias = resultado.get("referencias", {})
    linhas.append(f"- Total: **{referencias.get('total', 'n/d')}**; esperado: 90.")
    linhas.append(f"- Distribuição: `{referencias.get('por_token', {})}`.")
    linhas.append(f"- Duplicatas normalizadas: **{len(referencias.get('duplicatas_normalizadas', []))}**.")
    linhas.extend(["", "## Perguntas de avaliação", ""])
    avaliacao = resultado.get("avaliacao", {})
    linhas.append(f"- Total: **{avaliacao.get('total', 'n/d')}**; esperado: 100.")
    linhas.append(f"- Distribuição: `{avaliacao.get('por_token', {})}`.")
    linhas.append(f"- Duplicatas normalizadas: **{len(avaliacao.get('duplicatas_normalizadas', []))}**.")
    comuns = resultado.get("frases_idênticas_referencia_avaliacao", [])
    linhas.append(f"- Frases idênticas entre referências e avaliação: **{len(comuns)}**.")
    particoes = resultado.get("particoes", {})
    linhas.extend(["", "## Divisão validação/teste", ""])
    linhas.append(f"- Semente: **{particoes.get('random_state', RANDOM_STATE)}**; validação: **{particoes.get('validation_size', VALIDATION_SIZE):.0%}**.")
    linhas.append(f"- Tamanhos: validação **{particoes.get('total_validacao', 'n/d')}**, teste **{particoes.get('total_teste', 'n/d')}**.")
    linhas.append(f"- Sobreposição normalizada: **{len(particoes.get('sobreposicao_normalizada', []))}**.")
    linhas.append(f"- Reprodução determinística: **{'sim' if particoes.get('reproducao_deterministica') else 'não'}**.")
    linhas.append(f"- Compatibilidade dos arquivos gerados: `{resultado.get('arquivos_gerados', {})}`.")
    linhas.extend(["", "## Candidatos a revisão de ambiguidade", "", "Os itens abaixo foram encontrados por termos de objetos relacionados. Eles são apenas alertas; não constituem decisão de anotação.", ""])
    if resultado.get("ambiguidades_candidatas"):
        linhas.extend(["| Pergunta | Rótulo | Tokens sugeridos por termos |", "|---|---|---|"])
        for item in resultado["ambiguidades_candidatas"]:
            linhas.append(f"| {item['frase']} | `{item['token_esperado']}` | {', '.join(item['tokens_sugeridos_por_termos'])} |")
    else:
        linhas.append("Nenhum candidato foi identificado pelos critérios definidos.")
    linhas.extend(["", "## Pendências", ""])
    if resultado["problemas"]:
        linhas.extend(f"- {problema}" for problema in resultado["problemas"])
    else:
        linhas.append("Nenhuma inconsistência automática encontrada.")
    linhas.append("")
    return "\n".join(linhas)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Raiz do projeto.")
    parser.add_argument("--saida", type=Path, default=None, help="Diretório dos artefatos da Semana 5.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    resultado = executar_validacao(args.root.resolve(), args.saida.resolve() if args.saida else None)
    print(f"Relatório salvo em: {(args.saida or args.root / 'results' / 'semana5').resolve()}")
    print(f"Integridade: {'APROVADA' if resultado['integridade_aprovada'] else 'COM PENDÊNCIAS'}")
    if resultado["problemas"]:
        for problema in resultado["problemas"]:
            print(f"- {problema}")
    return 0 if resultado["integridade_aprovada"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
