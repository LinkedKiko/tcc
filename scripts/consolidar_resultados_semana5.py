"""Consolida resultados históricos e a avaliação funcional da Semana 5.

Nenhum resultado histórico ou julgamento humano é sobrescrito. Os arquivos
produzidos em ``results/semana5`` são derivados, auditáveis e identificam as
fontes usadas.

Uso:
    python scripts/consolidar_resultados_semana5.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ALL_TOKENS  # noqa: E402


FINAL_DIR = ROOT / "results" / "semana5"
FUNCTIONAL_REQUIRED = (
    "id_avaliacao", "id_pergunta", "pergunta", "token_esperado", "subintencao",
    "criterio", "estado_jogo", "modo", "threshold", "token_previsto", "similaridade",
    "classificacao_correta", "dica_observada", "assunto_adequado", "respeita_progresso",
    "sem_spoiler", "dica_adequada", "tipo_erro", "observacoes",
)
TOKEN_ALIASES = {
    "TOKEN_SIMBOLOS", "TOKEN_LIVRO", "TOKEN_MURAL", "TOKEN_COFRE",
    "TOKEN_RELOGIO", "TOKEN_NAO_RELACIONADO",
}


def markdown_table(dataframe: pd.DataFrame) -> str:
    """Renderiza uma tabela Markdown sem depender do pacote tabulate."""
    if dataframe.empty:
        return "(sem registros)"
    colunas = [str(coluna) for coluna in dataframe.columns]
    linhas = ["| " + " | ".join(colunas) + " |", "| " + " | ".join("---" for _ in colunas) + " |"]
    for valores in dataframe.itertuples(index=False, name=None):
        formatados = []
        for valor in valores:
            texto = "" if pd.isna(valor) else str(valor)
            formatados.append(texto.replace("|", "\\|").replace("\n", "<br>"))
        linhas.append("| " + " | ".join(formatados) + " |")
    return "\n".join(linhas)


def normalizar(valor: Any) -> str:
    texto = "" if valor is None else str(valor)
    texto = unicodedata.normalize("NFKD", texto.casefold())
    return "".join(c for c in texto if not unicodedata.combining(c)).strip()


def sim(valor: Any) -> bool:
    return normalizar(valor) in {"sim", "yes", "true", "1"}


def numero(valor: Any) -> float | None:
    try:
        if pd.isna(valor):
            return None
        return float(valor)
    except (TypeError, ValueError):
        return None


def ler_csv(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, keep_default_na=False)


def categoria_erro(linha: pd.Series) -> str:
    esperado = str(linha["token_esperado"])
    previsto = str(linha["token_previsto"])
    pergunta = normalizar(linha["pergunta"])
    if esperado == previsto:
        if not sim(linha["dica_adequada"]):
            return "token correto e dica inadequada"
        return "sem erro de classificação"
    if esperado == "TOKEN_NAO_RELACIONADO" and previsto != esperado:
        return "aceitação indevida"
    if esperado != "TOKEN_NAO_RELACIONADO" and previsto == "TOKEN_NAO_RELACIONADO":
        return "rejeição indevida"

    termos = {
        "TOKEN_SIMBOLOS": ("simbolo", "símbolo", "parede", "porta", "sequencia", "sequência", "painel", "ordem"),
        "TOKEN_LIVRO": ("livro", "enciclopedia", "enciclopédia", "pagina", "página", "traducao", "tradução", "correspondencia", "correspondência"),
        "TOKEN_MURAL": ("mural", "animal", "animais", "bicho", "criatura", "figura", "quantidade", "contar", "imagem"),
        "TOKEN_COFRE": ("cofre", "caixa", "cadeado", "compartimento", "fotografia", "senha", "codigo", "código", "combinacao", "combinação"),
        "TOKEN_RELOGIO": ("relogio", "relógio", "ponteiro", "horario", "horário", "hora", "minuto", "segundo"),
    }
    encontrados = [
        token for token, palavras in termos.items()
        if any(normalizar(palavra) in pergunta for palavra in palavras)
    ]
    if esperado in encontrados and previsto in encontrados:
        return "sobreposição semântica entre pistas"
    if not encontrados:
        return "formulação vaga ou indireta"
    return "confusão entre objetos relacionados"


def consolidar_comparacao(results: Path, saida: Path) -> tuple[pd.DataFrame, str]:
    origem = results / "comparacao_metricas.csv"
    dados = ler_csv(origem)
    required = {"metodo", "conjunto", "threshold", "total", "acertos", "erros", "acuracia", "macro_precision", "macro_recall", "macro_f1"}
    faltantes = sorted(required - set(dados.columns))
    if faltantes:
        raise ValueError(f"Métricas históricas sem colunas: {faltantes}")

    descricoes = {
        "Palavras-chave": {
            "classes_utilizadas": ", ".join(ALL_TOKENS),
            "politica_rejeicao": "TOKEN_NAO_RELACIONADO quando nenhuma palavra-chave foi encontrada; empate determinístico.",
            "arquivo_origem": "results/comparacao_metricas.csv",
        },
        "Embeddings sem threshold": {
            "classes_utilizadas": ", ".join(ALL_TOKENS),
            "politica_rejeicao": "Modo padrão: comparação com as 90 referências, inclusive TOKEN_NAO_RELACIONADO; sem threshold.",
            "arquivo_origem": "results/comparacao_metricas.csv; results/embeddings_teste_detalhado.csv",
        },
        "Embeddings com threshold 0.50": {
            "classes_utilizadas": ", ".join(ALL_TOKENS),
            "politica_rejeicao": "Modo rejeição: comparação somente com as 75 referências das cinco intenções; abaixo de 0,50 retorna TOKEN_NAO_RELACIONADO.",
            "arquivo_origem": "results/comparacao_metricas.csv; results/embeddings_threshold_teste_detalhado.csv; results/threshold_selecionado.csv",
        },
    }
    linhas = []
    for _, linha in dados.iterrows():
        extra = descricoes.get(str(linha["metodo"]), {
            "classes_utilizadas": ", ".join(ALL_TOKENS),
            "politica_rejeicao": "Não documentada no registro histórico.",
            "arquivo_origem": "results/comparacao_metricas.csv",
        })
        registro = linha.to_dict()
        registro.update(extra)
        linhas.append(registro)
    consolidado = pd.DataFrame(linhas)
    caminho = saida / "comparacao_final.csv"
    consolidado.to_csv(caminho, index=False, encoding="utf-8")

    esperado = {
        "Palavras-chave": (34, 0.5666666666666667, 0.5904431610700961),
        "Embeddings sem threshold": (40, 0.6666666666666666, 0.671173030615755),
        "Embeddings com threshold 0.50": (46, 0.7666666666666667, 0.766308462008793),
    }
    comparacao_linhas = []
    for _, linha in consolidado.iterrows():
        metodo = str(linha["metodo"])
        ref = esperado.get(metodo)
        coincidente = bool(ref and int(linha["acertos"]) == ref[0] and abs(float(linha["acuracia"]) - ref[1]) < 1e-9 and abs(float(linha["macro_f1"]) - ref[2]) < 1e-9)
        comparacao_linhas.append((metodo, coincidente))
    relatorio = gerar_relatorio_comparacao(consolidado, comparacao_linhas)
    (saida / "comparacao_final.md").write_text(relatorio, encoding="utf-8")
    return consolidado, str(origem)


def gerar_relatorio_comparacao(tabela: pd.DataFrame, coincidencias: list[tuple[str, bool]]) -> str:
    linhas = [
        "# Comparação final dos classificadores — Semana 5", "",
        "Os três cenários abaixo são os registros históricos da Semana 2. Nenhum PNG ou CSV histórico foi sobrescrito.", "",
        "## Tabela consolidada", "",
        markdown_table(tabela[["metodo", "conjunto", "threshold", "total", "acertos", "erros", "acuracia", "macro_precision", "macro_recall", "macro_f1"]]), "",
        "## Protocolos", "",
    ]
    for _, linha in tabela.iterrows():
        linhas.append(f"- **{linha['metodo']}**: classes `{linha['classes_utilizadas']}`; {linha['politica_rejeicao']} Fonte: `{linha['arquivo_origem']}`")
    linhas.extend(["", "## Conferência contra os valores históricos esperados", ""])
    for metodo, ok in coincidencias:
        linhas.append(f"- `{metodo}`: **{'coincidente' if ok else 'divergente'}**.")
    linhas.extend([
        "", "## Interpretação metodológica", "",
        "O cenário sem threshold e o cenário com threshold não são a mesma configuração com um único parâmetro alterado. O modo padrão usa as 90 referências, incluindo a classe não relacionada, enquanto o modo de rejeição usa somente 75 referências das cinco intenções e aplica o limiar de 0,50. Portanto, o ganho observado não é atribuído exclusivamente ao threshold.",
        "", "O threshold foi selecionado na validação pela maior Macro F1, com desempates por acurácia e menor threshold, e somente depois aplicado ao teste. Este relatório não reajusta parâmetros.", "",
    ])
    return "\n".join(linhas)


def consolidar_erros(results: Path, saida: Path) -> pd.DataFrame:
    origem = results / "embeddings_threshold_teste_erros.csv"
    erros = ler_csv(origem)
    obrigatorias = {"frase", "token_esperado", "token_previsto", "similaridade"}
    faltantes = sorted(obrigatorias - set(erros.columns))
    if faltantes:
        raise ValueError(f"Erros históricos sem colunas: {faltantes}")
    erros = erros.rename(columns={"frase": "pergunta"}).copy()
    # O arquivo histórico não traz a granularidade da análise funcional; a
    # categoria é uma classificação transparente para triagem, não um juízo
    # humano definitivo.
    erros["categoria_erro"] = erros.apply(
        lambda linha: categoria_erro(pd.Series({
            "token_esperado": linha["token_esperado"],
            "token_previsto": linha["token_previsto"],
            "pergunta": linha["pergunta"],
            "dica_adequada": "Não",
        })), axis=1,
    )
    erros["base_categorizacao"] = "triagem por rótulos e termos; requer revisão humana"
    colunas = ["pergunta", "token_esperado", "token_previsto", "similaridade", "categoria_erro", "base_categorizacao"]
    caminho = saida / "analise_erros.csv"
    erros[colunas].to_csv(caminho, index=False, encoding="utf-8")
    linhas = [
        "# Análise dos erros do melhor classificador — Semana 5", "",
        "Fonte: `results/embeddings_threshold_teste_erros.csv`. São 14 erros em 60 perguntas, preservados sem alteração.", "",
        "As categorias abaixo são uma triagem interpretável baseada nos tokens e em termos explícitos da pergunta. Não substituem a revisão humana de cada rótulo.", "",
        "## Distribuição por categoria", "",
        markdown_table(erros["categoria_erro"].value_counts().rename_axis("categoria").reset_index(name="quantidade")), "",
        "## Distribuição por token esperado", "",
        markdown_table(erros.groupby("token_esperado").size().rename("quantidade").reset_index()), "",
        "## Erros observados", "",
        markdown_table(erros[colunas]), "",
        "## Leitura por token", "",
        "Os erros envolvendo `TOKEN_COFRE`, `TOKEN_RELOGIO`, `TOKEN_MURAL`, `TOKEN_LIVRO` e `TOKEN_NAO_RELACIONADO` foram mantidos separados no CSV para permitir distinguir rejeições indevidas, aceitações indevidas e confusões entre intenções do puzzle. A matriz de confusão original permanece em `results/embeddings_threshold_teste_matriz_confusao.csv` e as métricas por token em `results/embeddings_threshold_teste_por_classe.csv`.", "",
        "A categoria `rejeição indevida` corresponde a pergunta legítima abaixo do limiar; `aceitação indevida` corresponde a entrada não relacionada aceita; os demais casos são confusões entre intenções relacionadas. Nenhuma frase de referência foi alterada.", "",
    ]
    (saida / "analise_erros.md").write_text("\n".join(linhas), encoding="utf-8")
    return erros


def validar_funcional(dados: pd.DataFrame) -> dict[str, Any]:
    problemas: list[str] = []
    faltantes = sorted(set(FUNCTIONAL_REQUIRED) - set(dados.columns))
    if faltantes:
        problemas.append(f"Colunas ausentes: {faltantes}")
        return {"problemas": problemas}
    if dados["id_avaliacao"].duplicated().any():
        problemas.append("Identificadores de cenário duplicados.")
    if set(dados["token_esperado"]) - TOKEN_ALIASES or set(dados["token_previsto"]) - TOKEN_ALIASES:
        problemas.append("Há tokens fora do conjunto permitido.")
    scores = pd.to_numeric(dados["similaridade"], errors="coerce")
    if scores.isna().any() or ((scores < -1) | (scores > 1)).any():
        problemas.append("Há similaridades não numéricas ou fora de [-1, 1].")
    for coluna in ("classificacao_correta", "assunto_adequado", "respeita_progresso", "sem_spoiler", "dica_adequada"):
        valores = {normalizar(valor) for valor in dados[coluna]}
        if not valores <= {"sim", "nao", "não"}:
            problemas.append(f"Valores inconsistentes em {coluna}: {sorted(valores)}")
    consistentes = dados.apply(lambda linha: (str(linha["token_esperado"]) == str(linha["token_previsto"])) == sim(linha["classificacao_correta"]), axis=1)
    if not bool(consistentes.all()):
        problemas.append("classificacao_correta não coincide com token esperado/previsto em todos os registros.")
    if dados["dica_observada"].map(lambda valor: not bool(str(valor).strip())).any():
        problemas.append("Há cenários sem texto de dica observada.")
    estados_validos = {"inicial", "cofre_aberto", "fotografia_encontrada", "mural_completo", "porta_aberta"}
    if set(dados["estado_jogo"]) - estados_validos:
        problemas.append("Há estados de jogo desconhecidos.")
    return {"problemas": problemas, "consistentes": bool(consistentes.all())}


def metricas_funcionais(dados: pd.DataFrame) -> pd.DataFrame:
    classificacao = dados["classificacao_correta"].map(sim)
    adequada = dados["dica_adequada"].map(sim)
    progresso = dados["respeita_progresso"].map(sim)
    spoiler = dados["sem_spoiler"].map(sim)
    linhas: list[dict[str, Any]] = []

    def adicionar(nome: str, numerador: int, denominador: int, unidade: str, definicao: str) -> None:
        linhas.append({"metrica": nome, "numerador": int(numerador), "denominador": int(denominador), "proporcao": numerador / denominador if denominador else None, "unidade": unidade, "definicao": definicao})

    adicionar("classificacao_por_pergunta_distinta", int((dados.groupby("id_pergunta")["classificacao_correta"].apply(lambda serie: serie.map(sim).all())).sum()), int(dados["id_pergunta"].nunique()), "pergunta", "Uma pergunta é considerada correta somente se todas as suas ocorrências em estados apresentarem token correto.")
    adicionar("classificacao_por_cenario", int(classificacao.sum()), len(dados), "cenário", "Cada uma das 27 linhas é uma avaliação de estado; perguntas repetidas não são independentes.")
    adicionar("dicas_adequadas", int(adequada.sum()), len(dados), "cenário", "Julgamento registrado pelo avaliador humano na coluna dica_adequada.")
    adicionar("sucesso_completo", int((classificacao & adequada).sum()), len(dados), "cenário", "classificacao_correta == Sim e dica_adequada == Sim.")
    adicionar("respeito_ao_progresso", int(progresso.sum()), len(dados), "cenário", "Julgamento registrado em respeita_progresso.")
    adicionar("ausencia_de_spoiler", int(spoiler.sum()), len(dados), "cenário", "Julgamento registrado em sem_spoiler.")

    quadrantes = {
        "token_errado_dica_inadequada": (~classificacao & ~adequada),
        "token_errado_dica_adequada": (~classificacao & adequada),
        "token_correto_dica_inadequada": (classificacao & ~adequada),
        "token_correto_dica_adequada": (classificacao & adequada),
    }
    for nome, mascara in quadrantes.items():
        adicionar(nome, int(mascara.sum()), len(dados), "cenário", "Cruzamento entre o token classificado e o julgamento da dica.")
    return pd.DataFrame(linhas)


def consolidar_funcional(root: Path, saida: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    origem = root / "data" / "avaliacao_funcional_dicas_resultados.csv"
    if not origem.exists():
        texto = "# Avaliação funcional das dicas — Semana 5\n\nO arquivo `data/avaliacao_funcional_dicas_resultados.csv` não foi localizado. Nenhum resultado foi inventado; é necessário receber o CSV preenchido pelo responsável pelo projeto.\n"
        (saida / "avaliacao_dicas_relatorio.md").write_text(texto, encoding="utf-8")
        return None, None, str(origem)
    dados = ler_csv(origem)
    validacao = validar_funcional(dados)
    dados_saida = dados.copy()
    dados_saida["quadrante_resultado"] = dados_saida.apply(
        lambda linha: (
            "token correto e dica adequada" if sim(linha["classificacao_correta"]) and sim(linha["dica_adequada"]) else
            "token correto e dica inadequada" if sim(linha["classificacao_correta"]) else
            "token errado e dica adequada" if sim(linha["dica_adequada"]) else
            "token errado e dica inadequada"
        ), axis=1,
    )
    dados_saida.to_csv(saida / "avaliacao_dicas_por_cenario.csv", index=False, encoding="utf-8")
    metricas = metricas_funcionais(dados)
    metricas.to_csv(saida / "avaliacao_dicas_metricas.csv", index=False, encoding="utf-8")

    observacoes_especiais = []
    for identificador in ("AV014", "AV020"):
        ocorrencia = dados.loc[dados["id_avaliacao"] == identificador]
        if ocorrencia.empty:
            observacoes_especiais.append(f"{identificador}: não presente no CSV atual.")
        else:
            linha = ocorrencia.iloc[0]
            observacoes_especiais.append(
                f"{identificador}: assunto_adequado={linha['assunto_adequado']!r}, dica_adequada={linha['dica_adequada']!r}; julgamento preservado."
            )
    linhas = [
        "# Avaliação funcional das dicas — Semana 5", "",
        "Fonte: `data/avaliacao_funcional_dicas_resultados.csv`. O CSV original não foi sobrescrito; `avaliacao_dicas_por_cenario.csv` acrescenta apenas uma coluna derivada para análise.", "",
        f"**Registros:** {len(dados)} cenários; **perguntas distintas:** {dados['id_pergunta'].nunique()}; **problemas de consistência automática:** {len(validacao['problemas'])}.", "",
        "## Métricas", "", markdown_table(metricas), "",
        "As métricas por pergunta distinta tratam uma pergunta como correta apenas quando todas as suas repetições de estado estão corretas. A métrica por cenário mantém as 27 linhas e explicita a repetição.", "",
        "## Quadrantes de classificação e dica", "",
        "A dica útil não elimina um erro de token, e token correto não garante pertinência da dica. Os quatro quadrantes estão registrados em `avaliacao_dicas_metricas.csv`.", "",
        "## Casos AV014 e AV020", "", *[f"- {texto}" for texto in observacoes_especiais],
        "", "Os campos humanos foram preservados. Uma possível interpretação é que uma dica possa ser útil apesar do token errado; outra é exigir que a adequação global dependa também do assunto correto. O CSV atual permanece a autoridade do julgamento e qualquer revisão deve ser documentada pelo avaliador.", "",
        "## Consistência", "",
    ]
    if validacao["problemas"]:
        linhas.extend(f"- {problema}" for problema in validacao["problemas"])
    else:
        linhas.append("Nenhuma inconsistência estrutural ou de coerência automática foi encontrada.")
    (saida / "avaliacao_dicas_relatorio.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return dados_saida, metricas, str(origem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    results = root / "results"
    saida = (results / "semana5" if args.saida is None else args.saida).resolve()
    saida.mkdir(parents=True, exist_ok=True)

    tabela, _ = consolidar_comparacao(results, saida)
    erros = consolidar_erros(results, saida)
    cenarios, metricas, origem_funcional = consolidar_funcional(root, saida)
    manifesto = {
        "fontes": [
            "results/comparacao_metricas.csv",
            "results/embeddings_threshold_teste_erros.csv",
            "results/embeddings_threshold_teste_detalhado.csv",
            "data/avaliacao_funcional_dicas_resultados.csv",
        ],
        "saida": str(saida.relative_to(root)),
        "total_metodos": int(len(tabela)),
        "total_erros_melhor_classificador": int(len(erros)),
        "avaliacao_funcional_presente": cenarios is not None,
        "origem_avaliacao_funcional": origem_funcional,
    }
    (saida / "manifesto_consolidacao.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Consolidação salva em: {saida}")
    print(tabela[["metodo", "acertos", "total", "acuracia", "macro_f1"]].to_string(index=False))
    if metricas is not None:
        print(metricas[["metrica", "numerador", "denominador", "proporcao"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
