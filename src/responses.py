"""Dicas predefinidas associadas aos tokens de decisão."""


DICAS = {
    "TOKEN_SIMBOLOS": (
        "Observe a ordem dos símbolos ao lado da porta. Talvez seja necessário "
        "descobrir o valor numérico associado a cada um."
    ),
    "TOKEN_LIVRO": (
        "A enciclopédia pode ajudar a identificar quais animais estão associados "
        "aos símbolos da linguagem fictícia."
    ),
    "TOKEN_MURAL": (
        "Observe a quantidade de cada animal no mural. A parte ausente pode ser "
        "necessária para descobrir os valores corretos."
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
        "Não foi possível relacionar sua pergunta aos elementos reconhecidos "
        "do puzzle. Tente perguntar sobre algum objeto da sala."
    ),
}

# Variantes curtas, predefinidas e determinísticas. A seleção contextual não
# gera texto: escolhe uma destas mensagens conforme token, pergunta e estado.
DICAS_CONTEXTUAIS = {
    "TOKEN_COFRE": {
        "senha_fechado": (
            "A senha do cofre é obtida observando o relógio: use as horas, os "
            "minutos e os segundos para formar a combinação."
        ),
        "abertura_fechado": (
            "Para abrir o cofre, examine primeiro o relógio e derive dele a "
            "combinação que deve ser inserida no painel."
        ),
        "abertura_aberto": (
            "O cofre já foi aberto. O procedimento consiste em obter a combinação "
            "no relógio e inseri-la no painel."
        ),
        "conteudo_fechado": (
            "Para descobrir o conteúdo do cofre, primeiro encontre a combinação "
            "no relógio e abra o painel."
        ),
        "finalidade_fechado": (
            "O cofre serve para guardar uma pista importante para a progressão; "
            "investigue o relógio antes de tentar acessá-lo."
        ),
        "finalidade": (
            "O cofre serve para guardar uma pista importante para a progressão; "
            "investigue o relógio antes de tentar acessá-lo."
        ),
        "finalidade_aberto": (
            "O cofre serve para guardar a fotografia que completa a pista do mural; "
            "a combinação para acessá-lo vem do relógio."
        ),
        "aberto": (
            "O cofre já está aberto. Procure a fotografia em seu interior e use-a "
            "para completar a parte ausente do mural."
        ),
        "fotografia": (
            "A fotografia já foi encontrada no cofre. Examine-a e use a parte "
            "ausente para completar o mural."
        ),
        "mural": (
            "A fotografia do cofre deve ser combinada com o mural para revelar "
            "as quantidades completas dos animais."
        ),
    },
    "TOKEN_RELOGIO": {
        "combinacao": (
            "O relógio fornece a pista para a combinação do cofre: observe as horas, "
            "os minutos e os segundos para formar a combinação."
        ),
        "observacao": (
            "Observe cuidadosamente as horas, os minutos e os segundos indicados "
            "pelo relógio e o que os ponteiros mostram."
        ),
        "consumida": (
            "Observe o horário e os ponteiros do relógio para entender a pista; "
            "a relação dele com o cofre continua sendo a combinação."
        ),
    },
    "TOKEN_MURAL": {
        # Antes de o cofre ser aberto, a dica fala apenas da investigação da
        # sala. Ela não confirma a existência, a localização ou a função da
        # fotografia.
        "incompleto": (
            "Observe os animais que aparecem na parte visível do mural. Como uma "
            "região está danificada, investigue os objetos da sala para reconstruir "
            "as informações ausentes."
        ),
        "incompleto_cofre_aberto": (
            "O mural ainda está incompleto. Examine cuidadosamente as pistas "
            "disponíveis no cofre para recuperar as informações ausentes."
        ),
        "fotografia_indisponivel": (
            "Essa informação ainda não está disponível. Observe a parte visível do "
            "mural e investigue os objetos acessíveis para avançar."
        ),
        "fotografia_cofre_aberto": (
            "Agora que o cofre está aberto, examine seu interior para encontrar a "
            "informação que pode completar a parte danificada do mural."
        ),
        "fotografia": (
            "Compare os animais mostrados na fotografia com os que aparecem na "
            "parte visível do mural para determinar os totais."
        ),
        "completo": (
            "Com o mural completo, use os totais dos animais junto com a "
            "enciclopédia e a ordem dos símbolos."
        ),
        "contagem_incompleta": (
            "Conte os animais presentes na parte visível. Como o mural está "
            "incompleto, ainda será necessário recuperar as informações ausentes."
        ),
        "contagem_fotografia": (
            "Conte os animais visíveis e combine essas quantidades com as "
            "informações da fotografia para obter os totais."
        ),
        "contagem_completa": (
            "Com as quantidades totais dos animais, consulte a enciclopédia e "
            "relacione cada total à ordem dos símbolos."
        ),
        "animais": (
            "Observe quais animais aparecem no mural e relacione cada um à "
            "associação correspondente na enciclopédia."
        ),
        "valores_incompleto": (
            "Os valores dependem da quantidade de cada animal. Como o mural está "
            "incompleto, ainda é necessário recuperar as informações ausentes."
        ),
        "valores_fotografia": (
            "Os valores dependem dos totais: combine a parte visível do mural com "
            "as informações da fotografia."
        ),
        "valores_completo": (
            "Use as quantidades totais dos animais juntamente com as associações "
            "da enciclopédia para obter os valores usados no puzzle."
        ),
    },
    "TOKEN_LIVRO": {
        "associacao": (
            "Use a enciclopédia para relacionar cada símbolo ao animal "
            "correspondente e consulte o mural para obter os valores."
        ),
        "traducao": (
            "Consulte a enciclopédia para traduzir os símbolos e descobrir qual "
            "animal corresponde a cada um."
        ),
        "conteudo": (
            "A enciclopédia contém a correspondência entre os símbolos e os "
            "animais."
        ),
        "utilizacao": (
            "Use a enciclopédia para relacionar símbolos e animais; depois cruze "
            "essa informação com o mural para resolver a porta."
        ),
        "mural_completo": (
            "Com o mural completo, cruze as associações da enciclopédia com os "
            "totais dos animais e a sequência dos símbolos."
        ),
        "generica": (
            "A enciclopédia traduz os símbolos em animais. Combine essa relação "
            "com as quantidades do mural para resolver a porta."
        ),
    },
    "TOKEN_SIMBOLOS": {
        "ordem": (
            "Anote a ordem ou sequência dos símbolos na parede. Depois consulte "
            "a enciclopédia e o mural para descobrir o valor de cada posição."
        ),
        "significado": (
            "Para entender o significado dos símbolos, compare-os com as "
            "associações registradas na enciclopédia."
        ),
        "associacao": (
            "Relacione cada símbolo ao animal correspondente na enciclopédia e "
            "use o mural para obter seu valor."
        ),
        "utilizacao": (
            "A sequência de símbolos deve ser aplicada aos valores obtidos no "
            "mural para formar a combinação da porta."
        ),
        "mural_incompleto": (
            "A parede fornece a sequência dos símbolos; para obter os valores, "
            "consulte a enciclopédia e complete a contagem do mural."
        ),
        "mural_completo": (
            "Aplique a ordem dos símbolos da parede aos totais obtidos no mural "
            "e relacionados pela enciclopédia."
        ),
    },
}


def obter_dica(token: str) -> str:
    """Retorna a dica do token ou uma mensagem segura para token desconhecido."""
    return DICAS.get(token, "Não há uma dica cadastrada para este token.")
