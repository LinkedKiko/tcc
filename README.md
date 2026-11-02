Protótipo desenvolvido para o Trabalho de Conclusão de Curso **“Aplicação de embeddings semânticos na geração de tokens de decisão para esclarecimento de dúvidas em jogos de puzzle”**.

## Visão geral

O projeto investiga o uso de embeddings semânticos para interpretar perguntas em português durante um jogo de puzzle. O sistema compara a pergunta do jogador com frases de referência, identifica uma intenção e emite um token de decisão. Esse token seleciona uma dica previamente escrita para o contexto do jogo.

O protótipo executa localmente e não utiliza API paga, banco de dados, treinamento, fine-tuning ou geração livre de texto.

## Como o sistema funciona

```text
Pergunta do jogador
        ↓
Geração do embedding semântico
        ↓
Comparação com as frases de referência
        ↓
Seleção da intenção mais próxima
        ↓
Token de decisão
        ↓
Dica previamente cadastrada
```

O classificador usa o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` e similaridade de cosseno. As frases de referência são codificadas na inicialização e reutilizadas durante a execução.

O sistema oferece dois modos de decisão:

- **Modo padrão:** compara a pergunta com todas as referências e retorna a classe mais próxima.
- **Modo com rejeição:** compara somente com as intenções do puzzle e retorna `TOKEN_NAO_RELACIONADO` quando a maior similaridade fica abaixo do limiar configurado.

## O puzzle demonstrativo

A aplicação inclui o cenário **Sala dos Símbolos**, no qual o jogador examina objetos, relaciona pistas e solicita dicas em linguagem natural. A lógica da partida é separada do classificador: o classificador identifica a intenção, enquanto o `HintManager` escolhe uma dica controlada de acordo com a pergunta e o estado atual do jogo.

## Funcionalidades

- Classificação semântica de perguntas em português.
- Ranking das frases de referência mais próximas.
- Rejeição opcional de perguntas fora do domínio por limiar de similaridade.
- Dicas estáticas e contextuais, sem geração livre de texto.
- Jogo funcional no terminal.
- Interface web local desenvolvida com Streamlit.
- Baseline de comparação por palavras-chave.
- Avaliação com acurácia, precisão, recall, F1 e matrizes de confusão.
- Análise de limiares com separação entre validação e teste.
- Geração de CSVs, gráficos e relatórios de resultados.
- Scripts para integridade dos dados, reprodução, benchmark e registro de testes.

## Tokens de decisão

| Token | Intenção representada |
|---|---|
| `TOKEN_SIMBOLOS` | Símbolos, sequência e painel da porta |
| `TOKEN_LIVRO` | Enciclopédia e tradução da linguagem fictícia |
| `TOKEN_MURAL` | Mural, animais e contagens |
| `TOKEN_COFRE` | Cofre e seu conteúdo |
| `TOKEN_RELOGIO` | Relógio, ponteiros e horário |
| `TOKEN_NAO_RELACIONADO` | Pergunta sem relação suficiente com o puzzle |

## Tecnologias

- Python 3.10 ou superior
- Sentence Transformers
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- Streamlit
- pytest

As dependências de instalação estão em `requirements.txt`. As versões do ambiente experimental registrado estão em `requirements-lock.txt`.

## Requisitos

- Python 3.10 ou superior instalado.
- Acesso à internet no primeiro carregamento, caso o modelo ainda não esteja no cache local.

## Instalação

Clone o repositório e execute os seguintes comandos na raíz do projeto:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se a política do PowerShell impedir a ativação do ambiente, autorize scripts somente na sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Na primeira classificação, o Sentence Transformers pode baixar o modelo multilíngue e armazená-lo no cache do usuário. As execuções seguintes podem funcionar sem acesso à internet, desde que o modelo permaneça disponível localmente.

## Como executar

Todos os comandos desta seção devem ser executados na raiz do projeto, com o ambiente virtual ativado.

### Interface web

```powershell
python -m streamlit run streamlit_app.py
```

Acesse o endereço local exibido pelo Streamlit no terminal.


### Jogo no terminal

```powershell
python game_demo.py
```

Exemplo com modo de rejeição:

```powershell
python game_demo.py --threshold 0.50
```

Outras opções disponíveis:

```text
--config CAMINHO       arquivo JSON de configuração do puzzle
--referencia CAMINHO   arquivo CSV com as frases de referência
--top-k N              quantidade de referências mantidas no ranking
--threshold VALOR      limiar usado no modo de rejeição
```

### Protótipo do classificador

```powershell
python main.py
```

Para ativar a rejeição por limiar:

```powershell
python main.py --threshold 0.50
```

Para executar as 100 perguntas do conjunto manual completo:

```powershell
python main.py --test
```

> `main.py --test` reúne validação e teste. Para a avaliação experimental com partições separadas, use `evaluate.py`.

## Dados

| Arquivo | Conteúdo | Finalidade |
|---|---|---|
| `data/frases_referencia.csv` | Frases e tokens | Base de comparação do classificador |
| `data/teste_manual.csv` | Perguntas, tokens esperados e dificuldade | Conjunto de avaliação |
| `data/puzzle_config.json` | Fatos e regras do cenário | Configuração da Sala dos Símbolos |
| `data/avaliacao_funcional_dicas.csv` | Casos funcionais | Avaliação das dicas |
| `data/avaliacao_funcional_dicas_resultados.csv` | Registros dos casos funcionais | Evidências da avaliação das dicas |

O conjunto atual contém 90 frases de referência e 100 perguntas de avaliação. A divisão experimental utiliza 40 perguntas para validação e 60 para teste, com a seed `42`.

Os scripts de avaliação não devem modificar os arquivos-fonte em `data/`. Os resultados são salvos em `results/`.

## Avaliação experimental

O arquivo `evaluate.py` oferece os seguintes modos:

```powershell
# Classificador semântico
python evaluate.py --modo embeddings

# Baseline por palavras-chave
python evaluate.py --modo baseline

# Comparação entre os métodos
python evaluate.py --modo comparar

# Análise de limiares na validação
python evaluate.py --modo thresholds

# Processo completo, com artefatos e relatório
python evaluate.py --modo completo
```

Exemplos de opções adicionais:

```powershell
python evaluate.py --modo embeddings --conjunto validacao
python evaluate.py --modo thresholds --thresholds 0.3 0.4 0.5
python evaluate.py --modo completo --sem-graficos
python evaluate.py --modo completo --resultados results/minha_execucao
```

O modo completo seleciona o limiar somente no conjunto de validação e aplica o valor escolhido uma única vez ao conjunto de teste.

### Métricas

- Acurácia
- Precisão macro
- Recall macro
- Macro F1
- Métricas por classe
- Matriz de confusão
- Aceitações e rejeições no experimento com limiar

## Resultados

Resultados históricos consolidados no conjunto de teste com 60 perguntas:

| Método | Acurácia | Macro F1 |
|---|---:|---:|
| Baseline por palavras-chave | 56,67% | 0,5904 |
| Embeddings sem limiar | 66,67% | 0,6712 |
| Embeddings com rejeição (`0,50`) | 76,67% | 0,7663 |

## Testes automatizados

Execute a suíte completa:

```powershell
python -m pytest -q
```

Execute somente os testes que não dependem do modelo real:

```powershell
python -m pytest -m "not integracao"
```

Execute somente a integração com o modelo real:

```powershell
python -m pytest -m integracao
```


## Reprodutibilidade e benchmark

Valide a integridade dos dados e consolide os resultados:

```powershell
python scripts/validar_integridade_semana5.py
python scripts/consolidar_resultados_semana5.py
```

Execute a reprodução registrada da avaliação:

```powershell
python scripts/executar_reproducao_semana5.py
```

Execute o benchmark estratificado de latência:

```powershell
python scripts/benchmark_latencia.py --por-classe 5 --seed 42 --repeticoes 30 --aquecimentos 3
```

Registre as execuções finais dos testes:

```powershell
python scripts/registrar_testes_semana5.py
```

As evidências são armazenadas em `results/semana5/`.

## Estrutura do projeto

```text
.
├── data/                  # dados, referências e configuração do puzzle
├── docs/                  # metodologia, arquitetura e relatórios técnicos
├── results/               # métricas, gráficos, relatórios e evidências
├── screenshots/           # capturas da interface, quando disponíveis
├── scripts/               # integridade, reprodução, benchmark e registros
├── src/
│   ├── classifier.py      # classificação semântica
│   ├── embeddings.py      # carregamento do modelo e embeddings
│   ├── similarity.py      # similaridade e ranking
│   ├── keyword_classifier.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── plots.py
│   ├── reporting.py
│   ├── responses.py
│   └── game/              # estado, regras e dicas do puzzle
├── tests/                 # testes automatizados
├── main.py                # demonstração do classificador no terminal
├── game_demo.py           # jogo no terminal
├── streamlit_app.py       # interface web
├── evaluate.py            # avaliação experimental
├── requirements.txt
├── requirements-lock.txt
└── pytest.ini
```

## Arquitetura

A aplicação separa as responsabilidades principais:

- `PuzzleClassifier` transforma a pergunta em um token de decisão.
- `HintManager` seleciona uma dica cadastrada para o token e o contexto.
- `PuzzleEngine` controla o estado e as transições da partida.
- `main.py`, `game_demo.py` e `streamlit_app.py` fornecem formas diferentes de interação.
- `evaluate.py` executa os experimentos sem depender da interface do jogo.

```text
Interface ──→ PuzzleEngine ──→ PuzzleClassifier ──→ token
                  │                                  │
                  └──────────→ HintManager ←─────────┘
                                  │
                                  └──→ dica contextual
```

Consulte [`docs/arquitetura_puzzle.md`](docs/arquitetura_puzzle.md) para os contratos e as transições de estado.
