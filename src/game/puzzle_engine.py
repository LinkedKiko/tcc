"""Motor das regras do puzzle, independente de terminal ou web."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Mapping, Optional

from .hint_manager import HintManager
from .puzzle_config import PuzzleConfig, load_puzzle_config
from .puzzle_state import PuzzleState


OperationResult = dict[str, Any]


class PuzzleEngine:
    """Executa ações da sala e mantém o estado isolado de uma partida."""

    def __init__(
        self,
        config: PuzzleConfig | Path | str | Mapping[str, Any],
        state: Optional[PuzzleState] = None,
        hint_manager: Optional[HintManager] = None,
    ) -> None:
        if isinstance(config, PuzzleConfig):
            self.config = config
        elif isinstance(config, (Path, str)):
            self.config = load_puzzle_config(config)
        elif isinstance(config, Mapping):
            self.config = PuzzleConfig.from_dict(config)
        else:
            raise TypeError("config deve ser PuzzleConfig, caminho ou objeto de configuração.")
        if state is not None and not isinstance(state, PuzzleState):
            raise TypeError("state deve ser uma instância de PuzzleState ou None.")
        self.state = state if state is not None else PuzzleState()
        self.hint_manager = hint_manager or HintManager()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    def examinar_objeto(self, objeto: str) -> OperationResult:
        """Descreve um objeto sem alterar o estado da partida."""
        chave = _normalizar_objeto(objeto)
        if chave == "relogio" or "relogio" in chave:
            return self._resultado(
                True,
                f"O relógio está parado em {self.config.horario}.",
                "OBJETO_EXAMINADO",
            )
        if chave == "cofre" or "cofre" in chave:
            if self.state.cofre_aberto:
                mensagem = "O cofre está aberto. Há uma fotografia em seu interior."
            else:
                mensagem = "O cofre está fechado e possui um painel para inserir uma combinação numérica."
            return self._resultado(True, mensagem, "OBJETO_EXAMINADO")
        if chave == "fotografia" or "fotografia" in chave or "foto" == chave:
            if not self.state.foto_encontrada:
                return self._resultado(
                    False,
                    "Essa pista ainda não está disponível. Investigue os objetos "
                    "acessíveis da sala para descobrir como avançar.",
                    "FOTOGRAFIA_INDISPONIVEL",
                )
            fotografia = self._dados_fotografia()
            return self._resultado(
                True,
                "A fotografia mostra a parte ausente do mural: "
                + self._formatar_quantidades(fotografia, "fotografia")
                + ".",
                "OBJETO_EXAMINADO",
                dados=fotografia,
            )
        if chave == "mural" or "mural" in chave:
            if self.state.mural_completo:
                totais = self.config.totais_animais
                return self._resultado(
                    True,
                    "O mural está completo e permite identificar a quantidade total de cada animal. "
                    "Totais: "
                    + self._formatar_quantidades(totais, "total")
                    + ".",
                    "OBJETO_EXAMINADO",
                    dados={"totais": totais},
                )
            visiveis = self._quantidades_visiveis()
            return self._resultado(
                True,
                "Parte do mural está destruída, impedindo a contagem completa dos animais. "
                "Na parte visível há: "
                + self._formatar_quantidades(visiveis, "visível")
                + ".",
                "OBJETO_EXAMINADO",
                dados={"visiveis": visiveis},
            )
        if chave in {"enciclopedia", "livro"} or "enciclopedia" in chave:
            return self.consultar_enciclopedia()
        if chave in {"simbolos", "parede"} or "simbolo" in chave or "parede" in chave:
            return self._resultado(
                True,
                "Na parede há a sequência: " + " ".join(self.config.sequencia),
                "OBJETO_EXAMINADO",
                dados={"sequencia": list(self.config.sequencia)},
            )
        if chave == "porta" or "porta" in chave:
            if self.state.porta_aberta:
                mensagem = "A porta está aberta. Você venceu o puzzle e escapou da sala."
            else:
                mensagem = "A porta está fechada e possui um painel para inserir a sequência numérica final."
            return self._resultado(True, mensagem, "OBJETO_EXAMINADO")
        return self._resultado(False, f"Objeto não reconhecido: {objeto!r}.", "OBJETO_DESCONHECIDO")

    def abrir_cofre(self, senha: str) -> OperationResult:
        """Tenta abrir o cofre sem converter a senha para inteiro."""
        if self.state.cofre_aberto:
            return self._resultado(True, "O cofre já está aberto.", "COFRE_JA_ABERTO")
        if not isinstance(senha, str):
            return self._resultado(False, "A senha do cofre deve ser informada como texto.", "SENHA_INVALIDA")
        if senha != self.config.senha_cofre:
            return self._resultado(False, "A combinação do cofre está incorreta.", "SENHA_COFRE_INCORRETA")
        self.state.cofre_aberto = True
        return self._resultado(True, "O cofre foi aberto.", "COFRE_ABERTO")

    def encontrar_fotografia(self) -> OperationResult:
        """Retira a fotografia do cofre quando ele já está aberto."""
        if not self.state.cofre_aberto:
            return self._resultado(
                False,
                "O acesso a essa pista está bloqueado. Investigue os objetos "
                "disponíveis antes de tentar encontrá-la.",
                "FOTOGRAFIA_BLOQUEADA",
            )
        if self.state.foto_encontrada:
            return self._resultado(True, "A fotografia já foi encontrada.", "FOTOGRAFIA_JA_ENCONTRADA")
        self.state.foto_encontrada = True
        return self._resultado(
            True,
            "Você encontrou a fotografia que mostra a parte ausente do mural.",
            "FOTOGRAFIA_ENCONTRADA",
            dados=self._dados_fotografia(),
        )

    def completar_mural(self) -> OperationResult:
        """Completa o mural e calcula os totais a partir da configuração."""
        if not self.state.foto_encontrada:
            return self._resultado(False, "O mural ainda não pode ser completado. Investigue outras pistas da sala para recuperar as informações ausentes.", "MURAL_BLOQUEADO")
        if self.state.mural_completo:
            return self._resultado(True, "O mural já está completo.", "MURAL_JA_COMPLETO", dados={"totais": self.config.totais_animais})
        self.state.mural_completo = True
        return self._resultado(
            True,
            "O mural foi completado. Agora é possível contar todos os animais.",
            "MURAL_COMPLETO",
            dados={"totais": self.config.totais_animais},
        )

    def consultar_enciclopedia(self) -> OperationResult:
        """Apresenta somente a tradução símbolo → animal, sem revelar números."""
        associacoes = {
            simbolo: _nome_animal(animal)
            for simbolo, animal in self.config.enciclopedia.items()
        }
        linhas = [f"{simbolo} → {animal}" for simbolo, animal in associacoes.items()]
        return self._resultado(
            True,
            "A enciclopédia registra:\n" + "\n".join(linhas),
            "ENCICLOPEDIA_CONSULTADA",
            dados={"associacoes": associacoes},
        )

    def obter_senha_cofre(self) -> str:
        """Retorna a combinação derivada do relógio para uso programático."""
        return self.config.senha_cofre

    def calcular_senha_porta(self) -> str:
        """Calcula a senha final, sem armazená-la duplicadamente."""
        return self.config.senha_porta

    def calcular_senha_porta_resultado(self) -> OperationResult:
        """Expõe a senha derivada em formato padronizado para uma interface."""
        return self._resultado(
            True,
            "A senha da porta foi calculada a partir das pistas do puzzle.",
            "SENHA_PORTA_CALCULADA",
            dados={"senha": self.calcular_senha_porta()},
        )

    def abrir_porta(self, senha: str) -> OperationResult:
        """Compara a sequência informada com a senha derivada do puzzle."""
        if self.state.porta_aberta:
            return self._resultado(True, "A porta já está aberta. Você venceu o puzzle.", "VITORIA_JA_REGISTRADA")
        if not isinstance(senha, str):
            return self._resultado(False, "A senha da porta deve ser informada como texto.", "SENHA_INVALIDA")
        if senha != self.calcular_senha_porta():
            return self._resultado(False, "A sequência da porta está incorreta.", "SENHA_PORTA_INCORRETA")
        self.state.porta_aberta = True
        return self._resultado(True, "A porta foi aberta. Você venceu o puzzle e escapou da sala.", "VITORIA")

    def obter_estado(self) -> PuzzleState:
        """Retorna uma cópia do estado atual para evitar mutações externas."""
        return self.state.copiar()

    def reiniciar(self) -> OperationResult:
        """Reinicia a partida sem recarregar a configuração ou o modelo."""
        self.state.reiniciar()
        return self._resultado(True, "A partida foi reiniciada.", "JOGO_REINICIADO")

    # ------------------------------------------------------------------
    # Integração semântica
    # ------------------------------------------------------------------
    def processar_pergunta(
        self,
        pergunta: str,
        classificador: Any,
        modo: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> OperationResult:
        """Classifica uma pergunta e seleciona uma dica sem alterar o puzzle."""
        # O classificador recebe exatamente os mesmos argumentos públicos de
        # antes. O snapshot é usado somente depois da classificação, na camada
        # de seleção da dica, e nunca é armazenado nem alterado pelo gerenciador.
        resultado = self.hint_manager.processar_pergunta(
            pergunta,
            classificador,
            modo=modo,
            threshold=threshold,
            estado=self.obter_estado().to_dict(),
        )
        resultado["estado"] = self.obter_estado().to_dict()
        return resultado

    def responder_pergunta(
        self,
        pergunta: str,
        classificador: Any,
        modo: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> OperationResult:
        """Alias de ``processar_pergunta`` para consumidores da interface."""
        return self.processar_pergunta(pergunta, classificador, modo, threshold)

    # ------------------------------------------------------------------
    def _resultado(
        self,
        sucesso: bool,
        mensagem: str,
        evento: str,
        dados: Optional[Mapping[str, Any]] = None,
    ) -> OperationResult:
        resultado: OperationResult = {
            "sucesso": sucesso,
            "mensagem": mensagem,
            "evento": evento,
            "estado": self.obter_estado().to_dict(),
        }
        if dados is not None:
            resultado["dados"] = dict(dados)
        return resultado

    def _quantidades_visiveis(self) -> dict[str, int]:
        return {animal: valores.visivel for animal, valores in self.config.animais.items()}

    def _dados_fotografia(self) -> dict[str, dict[str, int]]:
        return {
            animal: {"fotografia": valores.fotografia}
            for animal, valores in self.config.animais.items()
        }

    @staticmethod
    def _formatar_quantidades(
        quantidades: Mapping[str, Any], campo: str
    ) -> str:
        partes = []
        for animal, valor in quantidades.items():
            numero = valor[campo] if isinstance(valor, Mapping) else valor
            partes.append(f"{numero} {animal}")
        return ", ".join(partes)


def _normalizar_objeto(objeto: str) -> str:
    if not isinstance(objeto, str):
        return ""
    texto = unicodedata.normalize("NFKD", objeto.casefold())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(texto.split())


def _nome_animal(animal: str) -> str:
    nomes = {
        "libelula": "Libélula",
        "coruja": "Coruja",
        "tartaruga": "Tartaruga",
        "raposa": "Raposa",
    }
    return nomes.get(animal, animal.replace("_", " ").capitalize())
