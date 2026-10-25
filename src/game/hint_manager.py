"""Mapeamento de tokens e seleção determinística de dicas contextuais."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Optional

from src.config import TOKEN_NAO_RELACIONADO
from src.responses import DICAS, DICAS_CONTEXTUAIS


_MENSAGEM_TOKEN_DESCONHECIDO = "Não há uma dica cadastrada para este token."


class HintManager:
    """Integra a classificação com dicas controladas e conscientes do contexto.

    A classificação continua sendo responsabilidade exclusiva do classificador.
    Este gerenciador usa somente o token final, o texto da pergunta e, quando
    fornecido, um snapshot somente-leitura do estado da partida.
    """

    def __init__(self, dicas: Optional[Mapping[str, str]] = None) -> None:
        fonte = DICAS if dicas is None else dicas
        self.dicas = dict(fonte)
        self._usar_contexto = dicas is None
        if not self.dicas:
            raise ValueError("É necessário informar ao menos uma dica.")

    def obter_dica(self, token: str) -> str:
        """Retorna a dica estática correspondente ou uma mensagem segura."""
        if not isinstance(token, str) or not token.strip():
            return _MENSAGEM_TOKEN_DESCONHECIDO
        return self.dicas.get(token, _MENSAGEM_TOKEN_DESCONHECIDO)

    def selecionar_dica(
        self,
        token: str,
        pergunta: str,
        estado: Optional[Mapping[str, Any] | Any] = None,
    ) -> str:
        """Seleciona uma variante predefinida usando token, pergunta e estado.

        O método nunca classifica novamente a pergunta e nunca altera ``estado``.
        Instâncias criadas com um mapa customizado mantêm o comportamento legado
        e retornam a dica customizada, sem aplicar variantes internas.
        """
        if not self._usar_contexto or token not in DICAS_CONTEXTUAIS:
            return self.obter_dica(token)

        texto = _normalizar_texto(pergunta)
        flags = _ler_estado(estado)
        if token == "TOKEN_COFRE":
            chave = _selecionar_cofre(texto, flags)
        elif token == "TOKEN_RELOGIO":
            chave = _selecionar_relogio(texto, flags)
        elif token == "TOKEN_MURAL":
            chave = _selecionar_mural(texto, flags)
        elif token == "TOKEN_LIVRO":
            chave = _selecionar_livro(texto, flags)
        else:  # TOKEN_SIMBOLOS
            chave = _selecionar_simbolos(texto, flags)

        return DICAS_CONTEXTUAIS[token].get(chave, self.obter_dica(token))

    def get_hint(self, token: str) -> str:
        """Alias em inglês para facilitar consumo futuro por uma interface."""
        return self.obter_dica(token)

    def processar_pergunta(
        self,
        pergunta: str,
        classificador: Any,
        modo: Optional[str] = None,
        threshold: Optional[float] = None,
        estado: Optional[Mapping[str, Any] | Any] = None,
    ) -> dict[str, Any]:
        """Classifica uma pergunta e associa o token final à sua dica contextual."""
        if classificador is None or not hasattr(classificador, "classificar"):
            raise TypeError("O classificador precisa possuir o método classificar().")

        argumentos: dict[str, Any] = {}
        if modo is not None:
            argumentos["modo"] = modo
        if threshold is not None:
            argumentos["threshold"] = threshold
        classificacao = classificador.classificar(pergunta, **argumentos)
        if not isinstance(classificacao, Mapping):
            raise TypeError("O classificador deve retornar um mapeamento com o campo token.")
        token = classificacao.get("token", TOKEN_NAO_RELACIONADO)
        dica = self.selecionar_dica(token, pergunta, estado)
        return {
            "sucesso": True,
            "mensagem": dica,
            "evento": "DICA_SELECIONADA",
            "token": token,
            "dica": dica,
            "classificacao": dict(classificacao),
        }

    def integrar_classificador(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias explícito para o fluxo de integração token → dica."""
        return self.processar_pergunta(*args, **kwargs)

    def process_question(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Alias em inglês para integração com consumidores futuros."""
        return self.processar_pergunta(*args, **kwargs)


def classificar_e_obter_dica(
    classificador: Any,
    pergunta: str,
    modo: Optional[str] = None,
    threshold: Optional[float] = None,
    estado: Optional[Mapping[str, Any] | Any] = None,
) -> dict[str, Any]:
    """Executa a integração com um gerenciador padrão de dicas."""
    return HintManager().processar_pergunta(
        pergunta,
        classificador,
        modo=modo,
        threshold=threshold,
        estado=estado,
    )


def _normalizar_texto(texto: Any) -> str:
    if not isinstance(texto, str):
        return ""
    normalizado = unicodedata.normalize("NFKD", texto.casefold())
    sem_acentos = "".join(
        caractere for caractere in normalizado if not unicodedata.combining(caractere)
    )
    return " ".join(sem_acentos.split())


def _contem(texto: str, termos: tuple[str, ...]) -> bool:
    """Verifica palavras/expressões inteiras, evitando falsos positivos."""
    return any(
        re.search(r"(?<!\w)" + re.escape(_normalizar_texto(termo)) + r"(?!\w)", texto)
        for termo in termos
    )


def _ler_estado(estado: Optional[Mapping[str, Any] | Any]) -> dict[str, bool]:
    """Lê um snapshot sem guardar referência nem modificar o objeto recebido."""
    def valor(chave: str) -> bool:
        if estado is None:
            return False
        if isinstance(estado, Mapping):
            return bool(estado.get(chave, False))
        return bool(getattr(estado, chave, False))

    return {
        "cofre_aberto": valor("cofre_aberto"),
        "foto_encontrada": valor("foto_encontrada"),
        "mural_completo": valor("mural_completo"),
        "porta_aberta": valor("porta_aberta"),
    }


def _selecionar_cofre(texto: str, estado: Mapping[str, bool]) -> str:
    """Identifica o assunto da pergunta antes de consultar o progresso."""
    finalidade = _contem(
        texto,
        ("para que serve", "finalidade", "funcao", "objetivo", "utilidade", "serve"),
    )
    conteudo = _contem(
        texto,
        ("dentro", "conteudo", "o que tem", "interior", "fotografia", "foto", "imagem"),
    )
    senha = _contem(
        texto,
        ("senha", "combinacao", "codigo", "numero", "numeros"),
    )
    abertura = _contem(
        texto,
        ("abrir", "abertura", "destrancar", "acessar", "acesso", "fechado"),
    )

    # A subintenção explícita vence qualquer fallback de progresso. O estado
    # só escolhe uma variante segura dentro do mesmo assunto.
    if finalidade:
        return "finalidade_aberto" if estado["cofre_aberto"] else "finalidade_fechado"
    if conteudo:
        if estado["mural_completo"]:
            return "mural"
        if estado["foto_encontrada"]:
            return "fotografia"
        return "aberto" if estado["cofre_aberto"] else "conteudo_fechado"
    if senha:
        return "senha_fechado"
    if abertura:
        return "abertura_aberto" if estado["cofre_aberto"] else "abertura_fechado"

    # Sem subintenção confiável, a próxima ação pode ser contextual.
    if estado["mural_completo"]:
        return "mural"
    if estado["foto_encontrada"]:
        return "fotografia"
    if estado["cofre_aberto"]:
        return "aberto"
    return "conteudo_fechado"


def _selecionar_relogio(texto: str, estado: Mapping[str, bool]) -> str:
    # A relação com o cofre permanece pertinente mesmo depois de consumida.
    if _contem(texto, ("cofre", "combinacao", "senha", "codigo", "abrir")):
        return "combinacao"
    if _contem(texto, ("horario", "hora", "minuto", "segundo", "ponteiro")):
        return "observacao"
    return "consumida" if estado["cofre_aberto"] else "observacao"


def _selecionar_mural(texto: str, estado: Mapping[str, bool]) -> str:
    """Escolhe a orientação do mural respeitando assunto e descoberta.

    A pergunta explícita continua tendo prioridade. O estado apenas escolhe a
    variante segura para aquela subintenção; em particular, nenhuma pergunta
    inicial confirma a fotografia ou sua localização.
    """
    pergunta_fotografia = _contem(
        texto, ("fotografia", "foto", "parte ausente", "parte faltante")
    )
    pergunta_animais = _contem(
        texto,
        ("quais animais", "animais representados", "animais aparecem", "identificar os animais"),
    )
    pergunta_valores = _contem(
        texto, ("valor do animal", "valores dos animais", "valor numerico")
    )
    pergunta_contagem = _contem(
        texto, ("contar", "quantidade", "quantos", "numero", "numeros", "soma", "animais")
    )

    if pergunta_fotografia:
        if estado["mural_completo"]:
            return "completo"
        if estado["foto_encontrada"]:
            return "fotografia"
        if estado["cofre_aberto"]:
            return "fotografia_cofre_aberto"
        return "fotografia_indisponivel"
    if pergunta_animais:
        return "animais"
    if pergunta_valores:
        if estado["mural_completo"]:
            return "valores_completo"
        return "valores_fotografia" if estado["foto_encontrada"] else "valores_incompleto"
    if pergunta_contagem:
        if estado["mural_completo"]:
            return "contagem_completa"
        return "contagem_fotografia" if estado["foto_encontrada"] else "contagem_incompleta"
    if estado["mural_completo"]:
        return "completo"
    if estado["foto_encontrada"]:
        return "fotografia"
    if estado["cofre_aberto"]:
        return "incompleto_cofre_aberto"
    return "incompleto"


def _selecionar_livro(texto: str, estado: Mapping[str, bool]) -> str:
    if _contem(texto, ("para que serve", "como usar", "como utilizo", "utilizar", "resolver a porta", "abrir a porta")):
        return "utilizacao"
    if _contem(texto, ("traduzir", "traducao", "significado")):
        return "traducao"
    if _contem(texto, ("associacao", "correspondencia", "relacionar", "qual animal", "simbolo", "animal")):
        return "associacao"
    if _contem(texto, ("conteudo", "o que contem", "informacao", "pagina", "enciclopedia")):
        return "conteudo"
    if estado["mural_completo"]:
        return "mural_completo"
    return "generica"


def _selecionar_simbolos(texto: str, estado: Mapping[str, bool]) -> str:
    if _contem(texto, ("como uso", "como utilizar", "utilizacao")):
        return "utilizacao"
    if _contem(texto, ("ordem", "sequencia")):
        return "ordem"
    if _contem(texto, ("significado", "significam", "o que representam", "representam", "querem dizer")):
        return "significado"
    if _contem(texto, ("associar", "associacao", "correspondencia", "simbolo e numero", "valor de cada simbolo", "animal")):
        return "associacao"
    if _contem(texto, ("como uso", "como utilizar", "utilizacao", "porta", "abrir")):
        return "utilizacao"
    if _contem(texto, ("parede", "painel", "numero", "numeros")):
        return "ordem"
    return "mural_completo" if estado["mural_completo"] else "mural_incompleto"
