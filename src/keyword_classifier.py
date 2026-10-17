"""Classificador baseline por palavras-chave.

Este módulo não utiliza embeddings. Ele serve como linha de base experimental
para a comparação com o classificador semântico, permitindo verificar quanto
do desempenho pode ser obtido apenas com casamento lexical.

Procedência das listas
----------------------
O vocabulário foi definido a priori, a partir de duas fontes:

1. a descrição textual do puzzle (porta com painel numérico, símbolos na
   parede, enciclopédia de animais, mural quebrado, cofre e relógio parado);
2. o vocabulário já presente em ``data/frases_referencia.csv``, que é o
   conhecimento declarado do sistema.

As frases de ``data/teste_manual.csv`` não foram consultadas para montar ou
ajustar as listas, de modo que o baseline não é otimizado para o conjunto de
avaliação.

Palavras genéricas excluídas de propósito
-----------------------------------------
Os termos ``senha``, ``codigo``, ``combinacao``, ``numero``, ``animal``,
``pista``, ``puzzle``, ``sala``, ``abrir`` e ``tempo`` aparecem em frases de
referência de mais de uma intenção (ou em perguntas não relacionadas) e, por
isso, não foram incluídos em nenhuma lista: sua presença produziria atribuições
arbitrárias. ``tempo``, em particular, aparece tanto em contextos do relógio
quanto em perguntas sobre clima, que são não relacionadas.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import PUZZLE_TOKENS, TOKEN_NAO_RELACIONADO

# ---------------------------------------------------------------------------
# Vocabulário do baseline
# ---------------------------------------------------------------------------
# As chaves das entradas são comparadas sem acentos e em minúsculas. Entradas
# com espaço são tratadas como expressões contíguas (n-gramas).
PALAVRAS_CHAVE: Dict[str, Tuple[str, ...]] = {
    "TOKEN_SIMBOLOS": (
        "simbolo",
        "simbolos",
        "parede",
        "sequencia",
        "escrita",
        "escrito",
        "escritos",
        "traduzir",
        "desenho",
        "desenhos",
        "painel",
        "porta",
        "sinal",
        "sinais",
    ),
    "TOKEN_LIVRO": (
        "livro",
        "enciclopedia",
        "pagina",
        "paginas",
        "leitura",
        "dicionario",
        "volume",
        "obra",
        "folhear",
        "catalogo",
    ),
    "TOKEN_MURAL": (
        "mural",
        "contar",
        "quantidade",
        "quantidades",
        "quantos",
        "quantas",
        "figura",
        "figuras",
        "criatura",
        "criaturas",
        "especie",
        "especies",
        "libelula",
        "libelulas",
    ),
    "TOKEN_COFRE": (
        "cofre",
        "caixa",
        "cadeado",
        "compartimento",
        "guardar",
        "guardado",
        "guardada",
        "trancado",
        "trancada",
        "fotografia",
        "foto",
    ),
    "TOKEN_RELOGIO": (
        "relogio",
        "hora",
        "horas",
        "horario",
        "ponteiro",
        "ponteiros",
        "minuto",
        "minutos",
        "segundo",
        "segundos",
        "mostrador",
        "parado",
        "travado",
    ),
}

# Regra de desempate: entre os tokens com a maior contagem, vence o primeiro
# desta ordem. A ordem foi definida a priori pela cadeia de dependências do
# puzzle, percorrida do elemento mais específico para o mais genérico:
# o relógio abre o cofre, o cofre completa o mural, o mural dá valor aos
# símbolos e o livro faz a tradução. Os termos de TOKEN_SIMBOLOS ("parede",
# "sequencia", "escrita", "porta") são os mais compartilhados com as outras
# intenções nas frases de referência, por isso ficam em último lugar.
ORDEM_DESEMPATE: Tuple[str, ...] = (
    "TOKEN_RELOGIO",
    "TOKEN_COFRE",
    "TOKEN_LIVRO",
    "TOKEN_MURAL",
    "TOKEN_SIMBOLOS",
)

_PONTUACAO = re.compile(r"[^0-9a-z\s]+")


def remover_acentos(texto: str) -> str:
    """Remove os sinais diacríticos preservando as letras base."""
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(caractere for caractere in decomposto if not unicodedata.combining(caractere))


def normalizar(texto: str) -> List[str]:
    """Normaliza um texto e devolve a lista de palavras.

    Etapas: conversão para minúsculas, remoção de acentos, substituição de
    pontuação por espaço e separação por espaços em branco.

    >>> normalizar("Os SÍMBOLOS da parede, são pistas?")
    ['os', 'simbolos', 'da', 'parede', 'sao', 'pistas']
    """
    if not isinstance(texto, str):
        raise ValueError("O texto a normalizar precisa ser uma string.")
    minusculo = remover_acentos(texto.casefold())
    sem_pontuacao = _PONTUACAO.sub(" ", minusculo)
    return sem_pontuacao.split()


class KeywordClassifier:
    """Classifica perguntas contando correspondências com listas de palavras.

    A contagem considera palavras completas: ``"relogio"`` casa com
    ``"relogio"``, mas não com ``"relogioeiro"``. Cada palavra-chave distinta
    encontrada vale um ponto, independentemente de quantas vezes aparece na
    pergunta, para que a repetição de um mesmo termo não domine a decisão.
    """

    def __init__(
        self,
        palavras_chave: Mapping[str, Sequence[str]] = PALAVRAS_CHAVE,
        ordem_desempate: Sequence[str] = ORDEM_DESEMPATE,
        token_rejeicao: str = TOKEN_NAO_RELACIONADO,
    ) -> None:
        if not palavras_chave:
            raise ValueError("É necessário informar ao menos um token com palavras-chave.")

        self.token_rejeicao = token_rejeicao
        self.palavras_chave: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        for token, termos in palavras_chave.items():
            if not termos:
                raise ValueError(f"O token {token} não possui palavras-chave.")
            normalizados = []
            for termo in termos:
                partes = tuple(normalizar(termo))
                if not partes:
                    raise ValueError(f"Palavra-chave inválida no token {token}: {termo!r}")
                normalizados.append(partes)
            self.palavras_chave[token] = tuple(dict.fromkeys(normalizados))

        # Tokens sem ordem explícita vão para o fim, em ordem alfabética, de
        # modo que a decisão permaneça determinística mesmo se o vocabulário
        # for estendido.
        explicitos = [token for token in ordem_desempate if token in self.palavras_chave]
        restantes = sorted(set(self.palavras_chave) - set(explicitos))
        self.ordem_desempate: Tuple[str, ...] = tuple(explicitos + restantes)

    # ------------------------------------------------------------------
    def contar_correspondencias(self, pergunta: str) -> Dict[str, int]:
        """Conta, para cada token, quantas palavras-chave distintas casaram."""
        palavras = normalizar(_validar_pergunta(pergunta))
        presentes = set(palavras)
        contagens: Dict[str, int] = {}
        for token, termos in self.palavras_chave.items():
            total = 0
            for partes in termos:
                if len(partes) == 1:
                    if partes[0] in presentes:
                        total += 1
                elif _contem_sequencia(palavras, partes):
                    total += 1
            contagens[token] = total
        return contagens

    def classificar(self, pergunta: str) -> Dict[str, Any]:
        """Classifica uma pergunta e devolve o mesmo formato usado na avaliação.

        O campo ``similaridade`` é sempre ``None``: este método não calcula
        similaridade de cosseno e nenhum valor é inventado em seu lugar.
        """
        pergunta_limpa = _validar_pergunta(pergunta)
        contagens = self.contar_correspondencias(pergunta_limpa)
        maximo = max(contagens.values())

        if maximo == 0:
            token = self.token_rejeicao
            empatados: Tuple[str, ...] = ()
        else:
            empatados = tuple(
                nome for nome in self.ordem_desempate if contagens.get(nome, 0) == maximo
            )
            token = empatados[0]

        return {
            "pergunta": pergunta_limpa,
            "token": token,
            "similaridade": None,
            "contagens": contagens,
            "correspondencias": maximo,
            "empate": len(empatados) > 1,
            "tokens_empatados": empatados,
        }

    def classificar_em_lote(self, perguntas: Sequence[str]) -> List[Dict[str, Any]]:
        """Classifica uma sequência de perguntas."""
        return [self.classificar(pergunta) for pergunta in perguntas]


def _contem_sequencia(palavras: Sequence[str], alvo: Sequence[str]) -> bool:
    """Verifica se ``alvo`` aparece como sequência contígua em ``palavras``."""
    tamanho = len(alvo)
    if tamanho > len(palavras):
        return False
    alvo = list(alvo)
    return any(
        list(palavras[inicio : inicio + tamanho]) == alvo
        for inicio in range(len(palavras) - tamanho + 1)
    )


def _validar_pergunta(pergunta: Any) -> str:
    """Garante que a pergunta é um texto não vazio e devolve a versão limpa."""
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise ValueError("A pergunta não pode ser vazia.")
    return pergunta.strip()


# Tokens reconhecidos pelo baseline, úteis para validações externas.
TOKENS_BASELINE: Tuple[str, ...] = PUZZLE_TOKENS + (TOKEN_NAO_RELACIONADO,)
