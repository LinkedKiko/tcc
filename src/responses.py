"""Dicas associadas aos tokens de decisão."""

DICAS = {
    "TOKEN_SIMBOLOS": (
        "Observe a sequência de símbolos ao lado da porta e tente descobrir "
        "o que cada símbolo representa."
    ),
    "TOKEN_LIVRO": (
        "A enciclopédia pode ajudar a relacionar os símbolos da linguagem "
        "fictícia aos animais."
    ),
    "TOKEN_MURAL": (
        "Observe os animais presentes no mural e considere se a quantidade "
        "de cada um pode representar um valor."
    ),
    "TOKEN_COFRE": (
        "O conteúdo do cofre pode fornecer uma informação necessária para "
        "completar outra pista da sala."
    ),
    "TOKEN_RELOGIO": (
        "Observe cuidadosamente as horas, os minutos e os segundos indicados "
        "pelo relógio."
    ),
    "TOKEN_NAO_RELACIONADO": (
        "Essa pergunta não parece estar relacionada aos elementos do puzzle."
    ),
}


def obter_dica(token: str) -> str:
    """Retorna a dica do token ou uma mensagem segura para token desconhecido."""
    return DICAS.get(token, "Não há uma dica cadastrada para este token.")
