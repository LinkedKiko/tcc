# Arquitetura do puzzle — Semana 3

## 1. Objetivo da aplicação

A aplicação demonstra como perguntas em linguagem natural podem gerar tokens de decisão para selecionar dicas pré-cadastradas em um jogo de puzzle. O jogo da Sala dos Símbolos fornece um cenário controlado para conectar a classificação semântica das Semanas 1 e 2 a regras determinísticas de resolução, sem geração de texto por modelo.

## 2. Elementos do cenário

- **Relógio:** permanece parado em `09:15:30`; seus campos formam a combinação do cofre.
- **Cofre:** aceita a combinação derivada do relógio e contém a fotografia.
- **Fotografia:** representa a parte ausente do mural.
- **Mural:** combina a parte visível com a fotografia para calcular o total de cada animal.
- **Enciclopédia:** traduz cada símbolo fictício para um animal, sem informar seu número.
- **Símbolos:** a sequência junto à porta define a ordem dos valores obtidos no mural.
- **Porta:** aceita a sequência numérica final e registra a vitória.

A cadeia de resolução é relógio → cofre → fotografia → mural → enciclopédia/símbolos → porta. A enciclopédia pode ser consultada desde o início; a porta aceita uma senha correta mesmo que o jogador não tenha realizado todas as ações anteriores.

## 3. Regras de funcionamento

`data/puzzle_config.json` é a fonte centralizada. A combinação do cofre é derivada de `relogio.horario`, os totais do mural são `visivel + fotografia`, e a senha da porta percorre `parede.sequencia`, traduz cada símbolo pela `enciclopedia` e consulta o total do animal. Nenhuma dessas senhas é duplicada no JSON.

O cofre só muda para aberto com a string exata `091530`; zeros à esquerda não são descartados. A fotografia só pode ser encontrada depois disso. O mural só pode ser completado depois da fotografia. Consultas e dicas não alteram o estado. Tentativas repetidas são idempotentes e retornam eventos informativos.

## 4. Arquitetura de software

- **`PuzzleConfig`:** carrega e valida o JSON, inclusive horário, quantidades não negativas, símbolos, animais e sequência. Expõe valores derivados.
- **`PuzzleState`:** dataclass com apenas o estado da partida: `cofre_aberto`, `foto_encontrada`, `mural_completo` e `porta_aberta`.
- **`PuzzleEngine`:** aplica transições e retorna dicionários padronizados com `sucesso`, `mensagem`, `evento`, `dados` opcional e snapshot do estado. Não usa `input()` nem `print()`.
- **Classificador semântico:** `PuzzleClassifier` permanece responsável por embeddings, similaridade, top-k e threshold. Suas referências continuam pré-calculadas no construtor.
- **`HintManager`:** recebe o token final do classificador e seleciona uma dica estática de `src.responses`, sem depender de Sentence Transformers e sem mudar o estado.
- **`game_demo.py`:** contém somente a camada de terminal; constrói as dependências, lê comandos e apresenta resultados.

O fluxo de perguntas é `pergunta → PuzzleClassifier → token → HintManager → dica`. No modo de rejeição, o token final é usado: uma pergunta abaixo do threshold chega ao gerenciador como `TOKEN_NAO_RELACIONADO`.

## 5. Gerenciamento dos estados

Uma nova instância de `PuzzleEngine` cria um `PuzzleState` próprio. Abrir o cofre, encontrar a fotografia, completar o mural e abrir a porta são as transições que alteram flags. `obter_estado()` devolve uma cópia, e `reiniciar()` restaura as quatro flags para `False`. Não há banco de dados nem persistência.

## 6. Geração das senhas

- **Cofre:** remove os dois-pontos de `09:15:30`, produzindo `091530`.
- **Mural:** libélula `3 + 2 = 5`, coruja `1 + 1 = 2`, tartaruga `2 + 2 = 4` e raposa `1 + 2 = 3`.
- **Símbolos:** `◆ → libelula`, `☼ → coruja`, `△ → tartaruga`, `✦ → raposa`.
- **Porta:** a sequência `☼ ◆ ✦ △` resulta em `2 5 3 4`, portanto `2534`.

A implementação calcula esses valores sob demanda; alterar uma quantidade do JSON altera automaticamente os totais e a senha da porta.

## 7. Integração com tokens

`HintManager.processar_pergunta()` aceita qualquer objeto que exponha `classificar(pergunta)` e espera um mapeamento com `token`. Ele conserva a classificação completa em `classificacao`, escolhe uma única dica por token e retorna o resultado sem executar ações do puzzle. `PuzzleEngine.processar_pergunta()` acrescenta somente o estado observado, garantindo que perguntas não sejam ações.

Os seis tokens existentes não foram renomeados nem adicionados a `ALL_TOKENS`: `TOKEN_SIMBOLOS`, `TOKEN_LIVRO`, `TOKEN_MURAL`, `TOKEN_COFRE`, `TOKEN_RELOGIO` e `TOKEN_NAO_RELACIONADO`.

## 8. Testes

Os testes de `tests/test_puzzle_engine.py` cobrem carregamento, validação, horário, derivação das senhas, pré-condições, estados, idempotência, reinício, isolamento entre instâncias e zeros à esquerda. `tests/test_game_integration.py` usa um classificador falso para validar todas as dicas e comprovar que perguntas não mudam o estado.

Execute toda a suíte com:

```bash
python -m pytest -q
```

A camada de jogo não importa o modelo de embeddings. Assim, os testes das regras podem ser executados sem download e a mesma engine poderá ser reutilizada por uma futura interface Streamlit na Semana 4.
