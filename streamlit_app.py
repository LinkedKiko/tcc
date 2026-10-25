"""Interface web da Sala dos Símbolos.

Este módulo é somente uma camada de apresentação. As regras da partida,
as transições de estado e a seleção de dicas permanecem em ``src``.

Execução::

    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, MutableMapping, Optional

import streamlit as st

from src.config import DATASET_PATH, MODO_REJEICAO, PUZZLE_CONFIG_PATH

if TYPE_CHECKING:
    from src.classifier import PuzzleClassifier
else:
    # O classificador carrega dependências científicas pesadas apenas quando
    # uma pergunta é enviada. O atributo permanece disponível para injeção
    # nos testes e para manter a API interna da interface estável.
    PuzzleClassifier = None
from src.game import PuzzleConfig, PuzzleConfigError, PuzzleEngine

LOGGER = logging.getLogger(__name__)
INTERFACE_THRESHOLD = 0.50

ENGINE_KEY = "puzzle_engine"
OBJECT_RESULTS_KEY = "puzzle_object_results"
LAST_ACTION_KEY = "puzzle_last_action"
LAST_HINT_KEY = "puzzle_last_hint"
CLASSIFIER_ERROR_KEY = "puzzle_classifier_error"
VICTORY_ANIMATED_KEY = "puzzle_victory_animated"
INPUT_KEYS = ("cofre_senha", "porta_senha", "pergunta")

OBJECT_LABELS = {
    "relogio": "Relógio",
    "cofre": "Cofre",
    "fotografia": "Fotografia",
    "mural": "Mural",
    "enciclopedia": "Enciclopédia",
    "simbolos": "Símbolos da parede",
    "porta": "Porta",
}
ANIMAL_LABELS = {
    "libelula": "Libélula",
    "coruja": "Coruja",
    "tartaruga": "Tartaruga",
    "raposa": "Raposa",
}


@st.cache_resource(show_spinner=False)
def carregar_classificador(dataset_path: str = str(DATASET_PATH)) -> PuzzleClassifier:
    """Carrega o classificador uma vez e reutiliza-o entre reruns/sessões.

    ``PuzzleClassifier`` não mantém o progresso da partida: depois da
    construção, ele apenas consulta embeddings e referências. O estado mutável
    continua sendo mantido exclusivamente por uma engine em cada sessão.
    """
    classifier_type = PuzzleClassifier
    if classifier_type is None:
        from src.classifier import PuzzleClassifier as classifier_type

    return classifier_type(Path(dataset_path))


def criar_engine(config_path: Path | str = PUZZLE_CONFIG_PATH) -> PuzzleEngine:
    """Cria uma engine nova com a configuração validada do puzzle."""
    config = PuzzleConfig.from_file(config_path)
    return PuzzleEngine(config)


def obter_engine(
    session_state: MutableMapping[str, Any],
    config_path: Path | str = PUZZLE_CONFIG_PATH,
) -> PuzzleEngine:
    """Retorna a engine da sessão, sem recriá-la durante os reruns."""
    engine = session_state.get(ENGINE_KEY)
    if engine is None:
        engine = criar_engine(config_path)
        session_state[ENGINE_KEY] = engine
    return engine


def registrar_resultado(
    session_state: MutableMapping[str, Any],
    resultado: dict[str, Any],
    *,
    objeto: Optional[str] = None,
) -> None:
    """Guarda um retorno da engine para que ele sobreviva ao rerun."""
    if objeto is not None:
        resultados = session_state.setdefault(OBJECT_RESULTS_KEY, {})
        resultados[objeto] = resultado
    session_state[LAST_ACTION_KEY] = resultado


def executar_acao_objeto(
    engine: PuzzleEngine,
    session_state: MutableMapping[str, Any],
    objeto: str,
) -> dict[str, Any]:
    """Examina um objeto e registra o retorno sem duplicar regras do motor."""
    resultado = engine.examinar_objeto(objeto)
    registrar_resultado(session_state, resultado, objeto=objeto)
    return resultado


def executar_reinicio(
    engine: PuzzleEngine,
    session_state: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Reinicia a partida, as mensagens e os campos de entrada.

    Os widgets do Streamlit podem restaurar o último valor associado à chave
    mesmo quando ela é removida. Atribuir uma string vazia antes do rerun
    garante que os campos reapareçam limpos na nova renderização.
    """
    resultado = engine.reiniciar()
    session_state[OBJECT_RESULTS_KEY] = {}
    session_state[LAST_ACTION_KEY] = resultado
    session_state.pop(LAST_HINT_KEY, None)
    session_state.pop(CLASSIFIER_ERROR_KEY, None)
    session_state[VICTORY_ANIMATED_KEY] = False
    for key in INPUT_KEYS:
        session_state[key] = ""
    return resultado


def executar_pergunta(
    engine: PuzzleEngine,
    pergunta: str,
    classificador: Any,
) -> dict[str, Any]:
    """Envia uma pergunta pelo fluxo oficial engine → HintManager → classifier."""
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise ValueError("A pergunta não pode ser vazia.")
    return engine.processar_pergunta(
        pergunta.strip(),
        classificador,
        modo=MODO_REJEICAO,
        threshold=INTERFACE_THRESHOLD,
    )


def obter_estado_apresentacao(engine: PuzzleEngine) -> dict[str, bool]:
    """Obtém um snapshot serializável para os componentes visuais."""
    return engine.obter_estado().to_dict()


def _mostrar_resultado(resultado: Optional[dict[str, Any]]) -> None:
    if not resultado:
        return
    mensagem = resultado.get("mensagem", "")
    if resultado.get("sucesso"):
        st.success(mensagem)
    else:
        st.warning(mensagem)


def _mostrar_estado(engine: PuzzleEngine) -> None:
    estado = obter_estado_apresentacao(engine)
    st.sidebar.subheader("Estado da partida")
    itens = [
        ("Cofre", estado["cofre_aberto"], "Aberto", "Fechado"),
    ]
    # A fotografia só entra no inventário visual depois que o cofre foi aberto;
    # antes disso, até o nome do objeto é uma informação que o jogador precisa
    # descobrir durante a exploração.
    if estado["cofre_aberto"]:
        itens.append(("Fotografia", estado["foto_encontrada"], "Encontrada", "Disponível no cofre"))
    itens.extend(
        [
            ("Mural", estado["mural_completo"], "Completo", "Incompleto"),
            ("Porta", estado["porta_aberta"], "Aberta", "Fechada"),
        ]
    )
    for nome, ativo, positivo, negativo in itens:
        st.sidebar.write(f"**{nome}:** {positivo if ativo else negativo}")
    if estado["venceu"]:
        st.sidebar.success("Vitória registrada")


def _mostrar_detalhes_classificacao(resultado: dict[str, Any]) -> None:
    classificacao = resultado.get("classificacao", {})
    with st.expander("Detalhes da classificação"):
        st.write(f"**Token:** `{resultado.get('token', '—')}`")
        similaridade = classificacao.get("similaridade")
        if similaridade is not None:
            st.write(f"**Similaridade de cosseno:** {float(similaridade):.4f}")
        st.write(f"**Modo:** `{classificacao.get('modo', MODO_REJEICAO)}`")
        st.write(f"**Threshold:** {classificacao.get('threshold', INTERFACE_THRESHOLD):.2f}")
        st.write(f"**Frase de referência:** {classificacao.get('frase_referencia', '—')}")
        if classificacao.get("rejeitada"):
            st.info("A pergunta foi rejeitada por estar abaixo do threshold.")
        top_resultados = classificacao.get("top_resultados", [])
        if top_resultados:
            st.write("**Referências mais próximas:**")
            for posicao, item in enumerate(top_resultados[:3], start=1):
                st.write(
                    f"{posicao}. `{item.get('token', '—')}` — "
                    f"{float(item.get('similaridade', 0.0)):.4f} — {item.get('frase', '—')}"
                )
        st.caption("A similaridade é um escore do modelo, não uma probabilidade de acerto.")


def _mostrar_quantidades(dados: dict[str, Any], campo: str) -> None:
    """Renderiza dados que já foram calculados e devolvidos pela engine."""
    linhas = []
    for animal, valores in dados.items():
        valor = valores.get(campo) if isinstance(valores, dict) else valores
        if valor is not None:
            linhas.append({"Animal": ANIMAL_LABELS.get(animal, animal.replace("_", " ").capitalize()), "Quantidade": valor})
    if linhas:
        st.table(linhas)


def _resultado_do_objeto(session_state: MutableMapping[str, Any], objeto: str) -> Optional[dict[str, Any]]:
    return session_state.get(OBJECT_RESULTS_KEY, {}).get(objeto)


def _renderizar_relogio(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.markdown("#### 🕰️ Relógio")
    st.caption("Examine o relógio parado para observar a pista disponível.")
    if st.button("Examinar relógio", key="examinar_relogio"):
        executar_acao_objeto(engine, session_state, "relogio")
    _mostrar_resultado(_resultado_do_objeto(session_state, "relogio"))


def _renderizar_cofre(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.markdown("#### 🔒 Cofre")
    aberto = engine.obter_estado().cofre_aberto
    st.write(f"Estado: **{'aberto' if aberto else 'fechado'}**")
    with st.form("form_cofre", clear_on_submit=False):
        senha = st.text_input(
            "Combinação do cofre",
            key="cofre_senha",
            help="Digite a combinação como texto, preservando zeros à esquerda.",
        )
        tentar = st.form_submit_button("Tentar abrir o cofre")
    if tentar:
        if not senha.strip():
            st.warning("Informe uma combinação antes de tentar abrir o cofre.")
        else:
            resultado = engine.abrir_cofre(senha)
            registrar_resultado(session_state, resultado, objeto="cofre")
            if resultado.get("sucesso") and resultado.get("evento") == "COFRE_ABERTO":
                st.rerun()
    _mostrar_resultado(_resultado_do_objeto(session_state, "cofre"))
    if engine.obter_estado().cofre_aberto:
        if not engine.obter_estado().foto_encontrada:
            if st.button("Encontrar fotografia", key="encontrar_fotografia"):
                resultado = engine.encontrar_fotografia()
                registrar_resultado(session_state, resultado, objeto="fotografia")
                if resultado.get("sucesso") and resultado.get("evento") == "FOTOGRAFIA_ENCONTRADA":
                    st.rerun()
        else:
            st.caption("A fotografia já foi encontrada dentro do cofre.")


def _renderizar_fotografia(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    # A própria função permanece defensiva para que uma chamada acidental não
    # exponha o objeto antes de o cofre ser aberto.
    if not engine.obter_estado().cofre_aberto:
        return
    st.markdown("#### 🖼️ Fotografia")
    encontrada = engine.obter_estado().foto_encontrada
    st.write(f"Disponibilidade: **{'encontrada' if encontrada else 'indisponível'}**")
    if encontrada and st.button("Examinar fotografia", key="examinar_fotografia"):
        executar_acao_objeto(engine, session_state, "fotografia")
    resultado = _resultado_do_objeto(session_state, "fotografia")
    _mostrar_resultado(resultado)
    if resultado and resultado.get("sucesso"):
        dados = resultado.get("dados", {})
        _mostrar_quantidades(dados, "fotografia")


def _renderizar_mural(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.markdown("#### 🧩 Mural")
    estado = engine.obter_estado()
    st.write(f"Estado: **{'completo' if estado.mural_completo else 'incompleto'}**")
    if st.button("Examinar mural", key="examinar_mural"):
        executar_acao_objeto(engine, session_state, "mural")
    if estado.foto_encontrada and not estado.mural_completo:
        if st.button("Completar mural", key="completar_mural"):
            resultado = engine.completar_mural()
            registrar_resultado(session_state, resultado, objeto="mural")
            if resultado.get("sucesso") and resultado.get("evento") == "MURAL_COMPLETO":
                st.rerun()
    elif not estado.foto_encontrada:
        if estado.cofre_aberto:
            st.caption(
                "O mural está incompleto. Examine as pistas disponíveis no cofre "
                "para recuperar as informações da parte ausente."
            )
        else:
            st.caption(
                "O mural está incompleto. Investigue os objetos da sala para "
                "encontrar informações sobre a parte ausente."
            )
    else:
        st.caption("Compare as duas partes do mural para determinar as quantidades totais.")
    resultado = _resultado_do_objeto(session_state, "mural")
    _mostrar_resultado(resultado)
    if resultado and resultado.get("sucesso"):
        dados = resultado.get("dados", {})
        if "visiveis" in dados:
            _mostrar_quantidades(dados["visiveis"], "")
        if "totais" in dados:
            _mostrar_quantidades(dados["totais"], "")


def _renderizar_enciclopedia(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.markdown("#### 📖 Enciclopédia")
    st.caption("Consulte a associação entre os símbolos e os animais, sem os valores numéricos.")
    if st.button("Consultar enciclopédia", key="consultar_enciclopedia"):
        executar_acao_objeto(engine, session_state, "enciclopedia")
    resultado = _resultado_do_objeto(session_state, "enciclopedia")
    _mostrar_resultado(resultado)
    if resultado and resultado.get("sucesso"):
        st.table(
            [
                {"Símbolo": simbolo, "Animal": animal}
                for simbolo, animal in resultado.get("dados", {}).get("associacoes", {}).items()
            ]
        )


def _renderizar_simbolos(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.markdown("#### ✦ Símbolos da parede")
    st.caption("A ordem precisa ser relacionada aos animais e às quantidades do mural.")
    if st.button("Examinar símbolos", key="examinar_simbolos"):
        executar_acao_objeto(engine, session_state, "simbolos")
    resultado = _resultado_do_objeto(session_state, "simbolos")
    _mostrar_resultado(resultado)
    if resultado and resultado.get("sucesso"):
        sequencia = resultado.get("dados", {}).get("sequencia", [])
        if sequencia:
            st.markdown("### " + "  →  ".join(sequencia))


def _mostrar_vitoria(session_state: MutableMapping[str, Any], venceu: bool) -> None:
    """Exibe a vitória sempre, mas anima somente uma vez por sessão."""
    if not venceu:
        return
    if not session_state.get(VICTORY_ANIMATED_KEY, False):
        st.balloons()
        session_state[VICTORY_ANIMATED_KEY] = True
    st.success("Parabéns! Você concluiu a partida e escapou da Sala dos Símbolos.")


def _renderizar_porta(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.markdown("#### 🚪 Porta")
    aberta = engine.obter_estado().porta_aberta
    st.write(f"Estado: **{'aberta' if aberta else 'fechada'}**")
    with st.form("form_porta", clear_on_submit=False):
        senha = st.text_input(
            "Combinação da porta",
            key="porta_senha",
            help="Use a sequência deduzida a partir das pistas da sala.",
        )
        tentar = st.form_submit_button("Tentar abrir a porta")
    if tentar:
        if not senha.strip():
            st.warning("Informe uma combinação antes de tentar abrir a porta.")
        else:
            estado_antes = engine.obter_estado().porta_aberta
            resultado = engine.abrir_porta(senha)
            registrar_resultado(session_state, resultado, objeto="porta")
            if resultado.get("sucesso") and not estado_antes and engine.obter_estado().porta_aberta:
                session_state[VICTORY_ANIMATED_KEY] = False
                st.rerun()
    resultado = _resultado_do_objeto(session_state, "porta")
    _mostrar_resultado(resultado)
    _mostrar_vitoria(session_state, engine.obter_estado().venceu)


def _renderizar_perguntas(engine: PuzzleEngine, session_state: MutableMapping[str, Any]) -> None:
    st.header("Precisa de uma dica?")
    st.write(
        "Faça uma pergunta em português. O sistema classificará sua dúvida e "
        "selecionará uma dica cadastrada, sem gerar texto livre."
    )
    with st.form("form_pergunta", clear_on_submit=False):
        pergunta = st.text_area(
            "Sua pergunta",
            key="pergunta",
            placeholder="Ex.: O que representam os animais do mural?",
        )
        enviar = st.form_submit_button("Enviar pergunta")
    if enviar:
        if not pergunta.strip():
            st.warning("A pergunta não pode ser vazia.")
        else:
            try:
                with st.spinner("Carregando o modelo e classificando a pergunta..."):
                    classificador = carregar_classificador(str(DATASET_PATH))
                    resultado = executar_pergunta(engine, pergunta, classificador)
                session_state[LAST_HINT_KEY] = resultado
                session_state.pop(CLASSIFIER_ERROR_KEY, None)
            except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError):
                LOGGER.exception("Falha esperada ao carregar ou executar o classificador")
                session_state[CLASSIFIER_ERROR_KEY] = True
                session_state.pop(LAST_HINT_KEY, None)
            except Exception:  # pragma: no cover - última barreira da UI
                LOGGER.exception("Falha inesperada na classificação")
                session_state[CLASSIFIER_ERROR_KEY] = True
                session_state.pop(LAST_HINT_KEY, None)
    erro = session_state.get(CLASSIFIER_ERROR_KEY)
    if erro:
        st.error(
            "Não foi possível carregar ou executar o classificador. "
            "Verifique sentence-transformers, o modelo e o dataset de referência. "
            " Consulte os logs para obter detalhes técnicos."
        )
    resultado = session_state.get(LAST_HINT_KEY)
    if resultado:
        st.subheader("Dica")
        st.info(resultado.get("dica", "Não foi possível obter uma dica."))
        if resultado.get("token") == "TOKEN_NAO_RELACIONADO":
            st.warning("A pergunta não foi relacionada aos objetos reconhecidos do puzzle.")
        _mostrar_detalhes_classificacao(resultado)


def main() -> None:
    st.set_page_config(page_title="Sala dos Símbolos", page_icon="✦", layout="wide")
    st.title("✦ Sala dos Símbolos")
    st.write(
        "Você está em uma sala misteriosa. Observe os objetos, interprete as "
        "pistas e descubra a combinação necessária para abrir a porta."
    )
    st.info("Objetivo: explorar a sala, relacionar as pistas e concluir a partida sem revelar soluções automaticamente.")

    try:
        engine = obter_engine(st.session_state)
    except (FileNotFoundError, PuzzleConfigError, ValueError, OSError):
        LOGGER.exception("Falha ao iniciar a configuração do puzzle")
        st.error("Não foi possível iniciar a sala. Verifique a configuração do puzzle e consulte os logs.")
        st.stop()

    if st.sidebar.button("Reiniciar jogo", key="reiniciar_jogo"):
        executar_reinicio(engine, st.session_state)
        st.rerun()

    _mostrar_estado(engine)
    estado = obter_estado_apresentacao(engine)
    st.subheader(f"Porta: {'aberta — vitória registrada' if estado['porta_aberta'] else 'fechada'}")

    st.header("Objetos da sala")
    primeira_linha = st.columns(3)
    with primeira_linha[0]:
        _renderizar_relogio(engine, st.session_state)
    with primeira_linha[1]:
        _renderizar_cofre(engine, st.session_state)
    with primeira_linha[2]:
        if estado["cofre_aberto"]:
            _renderizar_fotografia(engine, st.session_state)

    segunda_linha = st.columns(3)
    with segunda_linha[0]:
        _renderizar_mural(engine, st.session_state)
    with segunda_linha[1]:
        _renderizar_enciclopedia(engine, st.session_state)
    with segunda_linha[2]:
        _renderizar_simbolos(engine, st.session_state)

    st.divider()
    _renderizar_porta(engine, st.session_state)
    st.divider()
    _renderizar_perguntas(engine, st.session_state)


if __name__ == "__main__":
    main()
