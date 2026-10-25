# Aplicação de embeddings semânticos em um puzzle

Protótipo e experimentos do TCC de Engenharia de Computação **"Aplicação de
embeddings semânticos na geração de tokens de decisão para esclarecimento de
dúvidas em jogos de puzzle"**.

Estado atual: Semana 4. O protótipo interativo das Semanas 1 e 2 e a lógica do
puzzle da Semana 3 continuam funcionando. A aplicação Streamlit acrescenta uma
camada web local para explorar a sala, resolver o puzzle e solicitar dicas em
linguagem natural, sem substituir os componentes experimentais.

---

## 1. Contexto e objetivo

O sistema recebe uma pergunta do jogador em português, gera seu embedding
semântico, compara esse vetor com os embeddings de frases de referência por
similaridade de cosseno e emite o **token de decisão** da frase mais próxima.
O token determina qual dica pré-escrita será apresentada.

O token é a unidade de decisão: o sistema não gera texto livre, apenas escolhe
entre dicas escritas pelo projetista do jogo. Isso mantém o conteúdo das dicas
sob controle e torna o comportamento auditável.

Tudo roda localmente. O projeto oferece o protótipo de terminal e uma interface
web local com Streamlit. Não usa banco de dados, banco vetorial, API paga, LLM
generativo, treinamento nem fine-tuning.

## 2. O puzzle

A sala contém cinco elementos encadeados:

1. **Porta com painel numérico** e uma sequência de símbolos na parede ao lado.
   Os símbolos representam os números a digitar.
2. **Enciclopédia de animais**, escrita na mesma linguagem fictícia dos
   símbolos. Permite relacionar símbolos a animais.
3. **Mural quebrado** com vários animais. A quantidade de cada tipo de animal
   revela o valor numérico do símbolo correspondente (cinco libélulas → o
   símbolo da libélula vale cinco).
4. **Cofre protegido por senha**, contendo uma fotografia da parte que falta no
   mural quebrado.
5. **Relógio parado**. As horas, minutos e segundos indicados pelos ponteiros
   formam a combinação do cofre.

A cadeia de dependências é: relógio → cofre → mural → símbolos → porta, com a
enciclopédia atuando como tradutor.

## 3. Tokens de decisão

| Token | Intenção da pergunta |
|---|---|
| `TOKEN_SIMBOLOS` | símbolos na parede, sequência, painel numérico da porta |
| `TOKEN_LIVRO` | enciclopédia de animais, tradução da linguagem fictícia |
| `TOKEN_MURAL` | mural quebrado, contagem dos animais, valores numéricos |
| `TOKEN_COFRE` | cofre, seu conteúdo, a fotografia que completa o mural |
| `TOKEN_RELOGIO` | relógio parado, ponteiros, horário como combinação |
| `TOKEN_NAO_RELACIONADO` | pergunta sem relação suficiente com o puzzle |

`TOKEN_NAO_RELACIONADO` merece atenção metodológica: perguntas irrelevantes não
formam uma categoria semanticamente homogênea. "Fora do escopo" não é um
assunto, mas o complemento de um conjunto de assuntos. Por isso o projeto
investiga duas formas distintas de tratá-lo (ver seção 10).

## 4. Arquitetura do classificador de embeddings

```
src/config.py           nome do modelo, caminhos, tokens, semente, grade de limiares
src/embeddings.py       carrega o SentenceTransformer uma vez e gera embeddings
src/similarity.py       similaridade de cosseno e ordenação determinística
src/classifier.py       PuzzleClassifier: carrega o CSV, pré-calcula referências, classifica
src/keyword_classifier.py  baseline por palavras-chave (sem embeddings)
src/dataset.py          carregamento e divisão determinística do conjunto de avaliação
src/evaluation.py       métricas, registro de erros, comparação, análise de limiares
src/plots.py            gráficos e matrizes de confusão em PNG
src/reporting.py        relatórios em Markdown
src/responses.py        mapeia token para dica
main.py                 protótipo interativo e teste manual
evaluate.py             experimentos da Semana 2
```

Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Sentence-BERT multilíngue, 384 dimensões).

Fluxo de uma classificação:

1. o modelo é carregado **uma única vez**, ao construir `PuzzleClassifier`;
2. as 90 frases de referência são codificadas nesse momento, em uma única
   chamada, resultando em uma matriz `90 × 384` mantida em memória;
3. para cada pergunta nova, **apenas o embedding da pergunta** é calculado;
4. a similaridade de cosseno é computada contra todas as referências;
5. o ranking é ordenado; empates são resolvidos pelo menor índice do CSV, o que
   torna o resultado reprodutível;
6. o token da referência mais próxima é emitido, junto com a similaridade e as
   três referências mais próximas (`TOP_K = 3`).

Nas avaliações, as perguntas de uma partição são codificadas em lote e a matriz
de similaridades é reaproveitada por todos os experimentos da execução,
inclusive pelos nove limiares da grade.

O programa trata perguntas vazias, dataset inexistente, dataset vazio, dataset
sem registros, colunas ausentes, frases vazias no CSV, falhas de leitura e
tokens desconhecidos, sempre com mensagens explicativas.

## 5. Baseline por palavras-chave

`src/keyword_classifier.py` implementa uma linha de base **sem embeddings**, que
dá sentido à comparação: sem ela, não é possível afirmar que o desempenho venha
da representação semântica e não de simples coincidência de palavras.

Funcionamento:

1. **normalização**: minúsculas, remoção de acentos (NFKD), pontuação
   substituída por espaço, separação em palavras;
2. **contagem**: para cada token, quantas palavras-chave distintas aparecem na
   pergunta. A comparação é por palavra completa, então "relogio" não casa com
   "relojoeiro". Repetir um termo vale um ponto;
3. **decisão**: vence o token com a maior contagem; sem nenhuma correspondência,
   o resultado é `TOKEN_NAO_RELACIONADO`;
4. **desempate** determinístico, na ordem fixa
   `TOKEN_RELOGIO → TOKEN_COFRE → TOKEN_LIVRO → TOKEN_MURAL → TOKEN_SIMBOLOS`,
   definida a priori pela cadeia de dependências do puzzle, do elemento mais
   específico para o mais genérico.

As listas foram montadas **a priori**, a partir da descrição do puzzle e do
vocabulário de `data/frases_referencia.csv`. As perguntas de avaliação não foram
consultadas, e as listas não foram revisadas após observar os resultados.

Termos genéricos como `senha`, `codigo`, `combinacao`, `numero`, `animal`,
`pista` e `tempo` foram **deliberadamente excluídos**: cada um aparece em
referências de mais de uma intenção, ou em perguntas não relacionadas, e
produziria atribuições arbitrárias.

O baseline não calcula similaridade de cosseno. Nos CSVs, o campo
`similaridade` fica **vazio**, sem valor inventado em seu lugar.

## 6. Organização dos datasets

```
data/frases_referencia.csv   90 frases, 6 tokens, 15 por token  (frase, token)
data/teste_manual.csv       100 perguntas  (frase, token_esperado, dificuldade)
```

Distribuição do conjunto de avaliação:

| Token esperado | Quantidade |
|---|---:|
| `TOKEN_SIMBOLOS` | 15 |
| `TOKEN_LIVRO` | 15 |
| `TOKEN_MURAL` | 15 |
| `TOKEN_COFRE` | 15 |
| `TOKEN_RELOGIO` | 15 |
| `TOKEN_NAO_RELACIONADO` | 25 |
| **Total** | **100** |

A coluna `dificuldade` indica a distância lexical até o vocabulário das frases
de referência, não a dificuldade de resolver o puzzle:

- `facil`: menciona explicitamente o objeto principal;
- `media`: usa sinônimos ou referências indiretas;
- `dificil`: expressa a intenção sem as palavras centrais das referências.

As cinco intenções do puzzle têm 5 perguntas de cada nível. Para
`TOKEN_NAO_RELACIONADO`, o nível mede a sobreposição lexical com o vocabulário
do jogo: 6 das 25 são **negativos difíceis**, que usam um termo do jogo
("livro", "horário", "senha", "sequência") com intenção externa.

## 7. Referência, validação e teste

Três conjuntos com funções distintas, que nunca se misturam:

| Conjunto | Arquivo | Tamanho | Função |
|---|---|---:|---|
| **Referência** | `data/frases_referencia.csv` | 90 | protótipos de intenção contra os quais toda pergunta é comparada |
| **Validação** | `results/particao_validacao.csv` | 40 | investigar limiares, comparar configurações, inspecionar problemas |
| **Teste** | `results/particao_teste.csv` | 60 | medir o desempenho final, com a configuração já fixada |

O conjunto de referência **não** é dado de treino: nenhum parâmetro é ajustado a
partir dele. Ele define o espaço de comparação. Consequência: o desempenho
medido avalia tanto o modelo de embeddings quanto a qualidade das 90 frases.

Validação e teste vêm da divisão de `data/teste_manual.csv`, estratificada por
token, com `train_test_split` e `RANDOM_STATE = 42`. A divisão é determinística:
execuções repetidas produzem exatamente as mesmas partições. O arquivo original
em `data/` nunca é modificado; as partições são apenas registradas em
`results/`.

Garantias verificadas automaticamente a cada execução:

- nenhuma frase de avaliação é idêntica a uma frase de referência;
- nenhuma frase aparece em validação e teste ao mesmo tempo;
- nenhuma frase de avaliação é adicionada às referências do classificador;
- **o limiar é escolhido apenas com dados de validação**.

## 8. Tecnologias

- Python 3.10 ou superior
- [sentence-transformers](https://www.sbert.net/) (traz `torch` e `transformers`)
- pandas, numpy
- scikit-learn (similaridade de cosseno, divisão estratificada, métricas)
- matplotlib (gráficos)
- pytest (testes)

## 9. Instalação

Na raiz do projeto, crie e ative um ambiente virtual.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, libere o script para a sessão atual:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Na primeira execução, `sentence-transformers` baixa o modelo multilíngue
(aproximadamente 470 MB) e o guarda em cache no diretório do usuário
(`~/.cache/huggingface`). É necessário acesso à internet apenas nesse primeiro
download; depois o projeto roda offline. Não há API paga envolvida.

Em máquinas sem GPU, instalar a versão CPU do PyTorch reduz bastante o download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 10. Execução do protótipo interativo

```bash
python main.py
```

Digite uma pergunta e pressione Enter. O programa mostra a frase de referência
mais próxima, o token identificado, a similaridade, o Top-3 e a dica
correspondente. Para encerrar, digite `sair`. Entradas vazias são informadas ao
usuário sem interromper o programa.

Rodar todas as 100 perguntas de avaliação e ver o relatório de métricas:

```bash
python main.py --test
```

Ativar o modo experimental de rejeição por limiar na demonstração:

```bash
python main.py --threshold 0.50
```

Nesse modo, perguntas cuja similaridade máxima frente às cinco intenções do
puzzle fique abaixo do limiar recebem `TOKEN_NAO_RELACIONADO`, e o programa
informa qual intenção foi descartada.

Opções adicionais: `--referencia CAMINHO` para outro CSV de referência e
`--top-k N` para alterar o tamanho do ranking exibido.

## 11. Execução das avaliações

Todos os comandos abaixo gravam seus resultados em `results/`.

```bash
# métricas do classificador semântico no conjunto de teste
python evaluate.py --modo embeddings

# métricas do baseline por palavras-chave
python evaluate.py --modo baseline

# os dois métodos sobre exatamente as mesmas perguntas, com análise de divergências
python evaluate.py --modo comparar

# grade de limiares sobre o conjunto de validação
python evaluate.py --modo thresholds

# sequência completa: divisão, limiares, avaliação no teste, gráficos e relatórios
python evaluate.py --modo completo
```

O modo `completo` executa, nesta ordem: carrega os datasets, cria a divisão
determinística, confere a integridade, carrega o classificador, analisa os
limiares **na validação**, seleciona um limiar, avalia os três cenários **no
teste**, compara os métodos, gera os CSVs, gera os gráficos e escreve o
relatório final.

Opções úteis:

```bash
--conjunto validacao        # usa a validação nos modos embeddings/baseline/comparar
--thresholds 0.3 0.4 0.5    # substitui a grade padrão de limiares
--sem-graficos              # pula a geração de PNGs no modo completo
--resultados OUTRO/DIR      # grava em outro diretório
```

O modelo é carregado no máximo uma vez por execução e nenhuma matriz de
similaridade é recalculada, de modo que a análise completa roda em um
computador pessoal em poucos segundos após o carregamento do modelo.

## 12. Testes automatizados

```bash
python -m pytest -q
```

Cobertura:

- `tests/test_classifier.py`: carregamento e validação do CSV, dataset
  inexistente, vazio e sem registros, colunas ausentes, frases vazias,
  perguntas vazias, estrutura do resultado, intervalo da similaridade,
  ordenação do Top-K, comportamento com e sem limiar, exclusão das referências
  não relacionadas no modo de rejeição, equivalência entre lote e chamadas
  individuais;
- `tests/test_keyword_classifier.py`: normalização de caixa, remoção de
  acentos, casamento por palavra completa, contagem de correspondências,
  ausência de correspondências, empate entre categorias, determinismo,
  pergunta vazia, retorno de `TOKEN_NAO_RELACIONADO`;
- `tests/test_evaluation.py`: cálculo de acurácia e das métricas por classe,
  tratamento de divisão por zero, quantidade de classificações, identificação
  de erros, geração dos CSVs, preservação dos rótulos, divisão determinística,
  ausência de sobreposição entre partições, seleção do limiar apenas na
  validação, contagens de aceitação e rejeição;
- `tests/test_integracao_modelo.py`: único arquivo que carrega o modelo real.

Os testes unitários **não baixam o modelo**: `tests/conftest.py` fornece um
serviço de embeddings determinístico de seis dimensões, e nenhum teste depende
de um score semântico exato para uma frase específica.

O teste de integração verifica apenas propriedades estruturais (dimensão dos
vetores, intervalo das similaridades, consistência entre lote e chamada
individual) e é pulado automaticamente se o modelo não puder ser carregado.

```bash
python -m pytest -m integracao        # somente o teste de integração
python -m pytest -m "not integracao"  # somente os testes unitários
```

## 13. O limiar de similaridade

Sem limiar, o classificador sempre devolve a referência mais próxima. Diante de
uma pergunta sem relação com o jogo, ele ainda aponta uma intenção, entregando
uma dica irrelevante. O limiar introduz a possibilidade de **abstenção**: se
nada é suficientemente próximo, o sistema não escolhe.

O projeto compara duas abordagens conceitualmente distintas.

**Abordagem A — classe no dataset de referência** (`MODO_PADRAO`)

`TOKEN_NAO_RELACIONADO` é uma classe como as outras, com 15 frases de
referência. A pergunta é comparada com todas as 90 referências e vence a maior
similaridade. É o comportamento da Semana 1, ativo quando
`SIMILARITY_THRESHOLD = None`.

**Abordagem B — rejeição por limiar** (`MODO_REJEICAO`)

Apenas as 75 referências das cinco intenções do puzzle participam da escolha.
`TOKEN_NAO_RELACIONADO` deixa de ser uma classe e passa a ser o resultado de uma
decisão:

```
pergunta → embedding → comparação com as 5 intenções → maior similaridade
    ≥ threshold  →  token da intenção
    <  threshold  →  TOKEN_NAO_RELACIONADO
```

A grade exploratória é `[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]`.
Esses valores **não** são limites teóricos: escalas de similaridade variam entre
modelos de embedding.

A escolha usa **somente o conjunto de validação**, por um critério
determinístico e documentado: maior Macro F1; em caso de empate, maior acurácia;
persistindo o empate, o menor limiar (rejeitar uma dúvida legítima prejudica
mais a experiência do que aceitar uma pergunta fora de contexto). O limiar
escolhido é então aplicado **uma única vez** ao conjunto de teste.

A análise distingue duas decisões diferentes: aceitar ou rejeitar, e acertar a
intenção. Uma pergunta relacionada aceita com o token errado conta como
aceitação correta, mas **não** como classificação correta. Sem essa distinção,
um limiar permissivo pareceria bom apenas por aceitar tudo.

`SIMILARITY_THRESHOLD` permanece `None` em `src/config.py`. O valor escolhido
experimentalmente é registrado em `results/`, não gravado no código.

## 14. Onde ficam os relatórios

Tudo em `results/`, gerado por execuções reais:

| Arquivo | Conteúdo |
|---|---|
| `relatorio_semana2.md` | relatório completo: dataset, métricas dos três cenários, limiares, erros e limitações |
| `comparacao_resumo.md` | leitura da comparação entre métodos |
| `particao_validacao.csv`, `particao_teste.csv`, `particao_resumo.csv` | composição das partições |
| `analise_thresholds.csv` | grade de limiares na validação |
| `threshold_selecionado.csv` | limiar escolhido e critério aplicado |
| `threshold_selecionado_teste.csv` | avaliação única no conjunto de teste |
| `comparacao_classificadores.csv` | predições dos dois métodos pergunta a pergunta |
| `comparacao_metricas.csv` | métricas agregadas dos cenários |
| `*_detalhado.csv` | predição, similaridade, acerto e dificuldade de cada pergunta |
| `*_erros.csv` | somente as perguntas classificadas incorretamente |
| `*_metricas.csv`, `*_por_classe.csv` | métricas agregadas e por token |
| `*_matriz_confusao.csv` | matriz de confusão em CSV |
| `grafico_comparacao_classificadores.png` | barras com acurácia e médias macro |
| `grafico_thresholds.png` | limiar versus desempenho (validação) |
| `grafico_distribuicao_similaridades.png` | similaridades de perguntas relacionadas e não relacionadas |
| `matriz_confusao_*.png` | matrizes de confusão dos três cenários |

A metodologia detalhada está em [`docs/metodologia_semana2.md`](docs/metodologia_semana2.md),
com as fórmulas das métricas, os critérios de construção do dataset e a
discussão das duas abordagens de rejeição.

## 15. Limitações da avaliação preliminar

Os números em `results/` são reais, mas preliminares. Delimitações honestas:

- **Dataset pequeno.** 90 frases de referência e 100 perguntas de avaliação. No
  conjunto de teste são 9 perguntas por intenção do puzzle e 15 não
  relacionadas; um único erro desloca o recall de uma classe em cerca de 11
  pontos percentuais. Diferenças pequenas entre métodos não são distinguíveis
  de variação amostral.
- **Rótulos não validados externamente.** As perguntas e seus rótulos foram
  elaborados pelo próprio autor. Não houve anotação independente nem medição de
  concordância entre avaliadores. **O conjunto deve ser tratado como uma
  proposta inicial de avaliação, sujeita a revisão humana, e não como um
  padrão-ouro.**
- **Ambiguidade semântica inerente.** As intenções são encadeadas por
  construção (o relógio abre o cofre, o cofre completa o mural). Perguntas que
  citam dois elementos admitem mais de uma leitura plausível, e o rótulo
  reflete a intenção julgada principal pelo autor.
- **Influência das frases de referência.** O classificador só reconhece uma
  intenção se houver uma frase de referência próxima. O resultado avalia
  conjuntamente o modelo e a escolha das 90 frases.
- **Sensibilidade do limiar.** Escolhido em uma grade de nove valores sobre 40
  perguntas de validação. Outro conjunto de validação poderia indicar outro
  valor, e o valor é específico deste modelo e deste conjunto de referência.
- **Sem jogadores reais.** Nenhuma pergunta foi coletada de pessoas jogando o
  puzzle. Dúvidas formuladas durante o jogo podem diferir das escritas em
  ambiente controlado.
- **Sem teste de significância.** O projeto não aplica testes estatísticos entre
  os métodos, e os relatórios não afirmam superioridade a partir de diferenças
  pequenas.

## 16. Estrutura do projeto

```text
.
├── data/
│   ├── frases_referencia.csv
│   ├── teste_manual.csv
│   └── puzzle_config.json
├── docs/
│   ├── metodologia_semana2.md
│   ├── arquitetura_puzzle.md
│   └── interface_semana4.md
├── screenshots/
│   └── README.md
├── results/
│   ├── .gitkeep
│   └── (CSVs, PNGs e relatórios gerados pelas avaliações)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── embeddings.py
│   ├── similarity.py
│   ├── classifier.py
│   ├── keyword_classifier.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── plots.py
│   ├── reporting.py
│   └── responses.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_classifier.py
│   ├── test_keyword_classifier.py
│   ├── test_evaluation.py
│   ├── test_streamlit_app.py
│   └── test_integracao_modelo.py
├── streamlit_app.py
├── main.py
├── evaluate.py
├── pytest.ini
├── requirements.txt
├── README.md
└── .gitignore
```

## 17. Formato dos CSVs

Referência (`data/frases_referencia.csv`):

```csv
frase,token
O que significam os símbolos escritos ao lado da porta?,TOKEN_SIMBOLOS
```

Avaliação (`data/teste_manual.csv`):

```csv
frase,token_esperado,dificuldade
Como interpretar os traços que aparecem na parede?,TOKEN_SIMBOLOS,facil
```

Resultado detalhado (`results/*_detalhado.csv`):

```csv
frase,token_esperado,token_previsto,similaridade,correto,dificuldade
```

Comparação (`results/comparacao_classificadores.csv`):

```csv
frase,token_esperado,token_keywords,token_embeddings,similaridade,acerto_keywords,acerto_embeddings,dificuldade
```

Análise de limiares (`results/analise_thresholds.csv`):

```csv
threshold,acuracia,macro_f1,aceitas_corretamente,rejeitadas_corretamente,falsas_aceitacoes,falsas_rejeicoes,...
```

Os arquivos em `data/` não são alterados por nenhum script. Toda escrita
acontece em `results/`.


## 16. Semana 3 — lógica do puzzle e demonstração de terminal

A Semana 3 adiciona uma partida funcional da **Sala dos Símbolos**, mantendo o classificador e os experimentos anteriores separados da lógica de resolução. A configuração está em `data/puzzle_config.json`; ela é validada por `PuzzleConfig` e contém somente os fatos do cenário. As senhas são derivadas: o relógio `09:15:30` produz a combinação do cofre `091530`, e a sequência `☼ ◆ ✦ △`, traduzida pela enciclopédia e pelos totais do mural, produz a senha da porta `2534`.

A cadeia do jogo é relógio → cofre → fotografia → mural → enciclopédia/símbolos → porta. O `PuzzleEngine` administra regras e estados sem chamar `input()` ou `print()`. `PuzzleState` é isolado por partida e pode ser reiniciado. `HintManager` recebe o token produzido pelo `PuzzleClassifier` e seleciona uma dica estática; perguntas e dicas não alteram o estado.

### Estrutura adicionada

```text
data/puzzle_config.json
src/game/__init__.py
src/game/puzzle_config.py
src/game/puzzle_state.py
src/game/puzzle_engine.py
src/game/hint_manager.py
game_demo.py
tests/test_puzzle_engine.py
tests/test_game_integration.py
docs/arquitetura_puzzle.md
```

### Executar o jogo

Com as dependências instaladas, execute:

```bash
python game_demo.py
```

O primeiro uso do classificador pode baixar o modelo multilíngue. Também é possível escolher arquivos e o modo experimental sem alterar a configuração da Semana 2:

```bash
python game_demo.py --config data/puzzle_config.json --referencia data/frases_referencia.csv
python game_demo.py --threshold 0.50
```

A demonstração permite examinar objetos, inserir as duas senhas como texto, obter a fotografia, completar o mural, consultar a enciclopédia e a sequência da parede, fazer perguntas, consultar o estado e reiniciar a partida. A lógica pode ser consumida diretamente por uma interface futura:

```python
from src.game import PuzzleEngine, load_puzzle_config

engine = PuzzleEngine(load_puzzle_config())
resultado = engine.abrir_cofre("091530")
```

A documentação dos contratos e transições está em [`docs/arquitetura_puzzle.md`](docs/arquitetura_puzzle.md). As avaliações da Semana 2 continuam sendo executadas pelos comandos descritos nas seções anteriores e não utilizam o JSON do puzzle.

## 18. Interface web da Semana 4

A interface gráfica é executada localmente com Streamlit e usa diretamente os componentes das semanas anteriores. Cada sessão do navegador mantém sua própria instância de `PuzzleEngine` em `st.session_state`; o classificador é carregado sob demanda quando a primeira pergunta é enviada e reutilizado por `st.cache_resource`. O progresso não é persistido em disco ou banco de dados.

### Instalação e execução

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

A dependência `streamlit` está em `requirements.txt`, junto com `sentence-transformers`, pandas, NumPy, scikit-learn, matplotlib e pytest. O primeiro uso de perguntas pode baixar o modelo multilíngue de embeddings e exige que `sentence-transformers` esteja instalado.

### Uso da interface

A tela **Sala dos Símbolos** apresenta seis objetos principais: relógio, cofre, mural, enciclopédia, símbolos da parede e porta. O jogador examina o relógio, abre o cofre como texto, encontra a fotografia, completa o mural, consulta a enciclopédia e os símbolos e tenta a combinação da porta. O fluxo da interface chama os métodos do `PuzzleEngine`; ela não duplica contagens, senhas ou regras do puzzle e não revela automaticamente as soluções.

A seção **Precisa de uma dica?** aceita perguntas em português. As perguntas são processadas pelo `PuzzleClassifier` por meio de `PuzzleEngine.processar_pergunta`, usando explicitamente `MODO_REJEICAO` e threshold `0.50`. A dica é escolhida pelo `HintManager`; o painel opcional **Detalhes da classificação** mostra token, escore, referências próximas, modo e threshold. O escore de similaridade não representa uma probabilidade de acerto. Perguntas fora do domínio recebem `TOKEN_NAO_RELACIONADO` e a mensagem cadastrada.

O botão **Reiniciar jogo** restaura o estado inicial, limpa a dica e os resultados transitórios, mas não recarrega o classificador. Para resolver a sala, observe as pistas e faça as deduções entre os objetos; as combinações não são exibidas como texto de solução na tela inicial.

### Testes e documentação

```powershell
python -m pytest -q
python -m pytest -m "not integracao"
python -m pytest -m integracao
```

O primeiro comando executa a suíte completa; o teste marcado como `integracao` depende do modelo real e pode ser pulado apenas quando a dependência ou o modelo estiverem indisponíveis. Os comandos de terminal e avaliação experimental continuam disponíveis:

```powershell
python main.py
python game_demo.py --threshold 0.50
python evaluate.py --modo completo
```

A arquitetura da interface, a estratégia de sessão, os testes e as limitações estão documentados em [`docs/interface_semana4.md`](docs/interface_semana4.md). As capturas reais produzidas durante a execução estão em `screenshots/`; o inventário e o roteiro de reprodução estão em [`screenshots/README.md`](screenshots/README.md).


## 18. Fechamento técnico da Semana 4

A interface da Semana 4 está em `streamlit_app.py`. A classificação de intenção e a seleção contextual são camadas distintas: o `PuzzleClassifier` usa embeddings para produzir um token; o `HintManager` não classifica novamente nem gera texto, apenas identifica uma subintenção por regras determinísticas controladas e escolhe uma dica cadastrada conforme o estado.

A prioridade da seleção é **pergunta explícita → subintenção → estado → variante segura**. Assim, uma pergunta sobre a senha do cofre continua relacionada ao relógio após o cofre ser aberto, uma pergunta sobre a contagem do mural continua tratando de contagem após o mural completo e uma pergunta sobre a ordem dos símbolos continua tratando de sequência após a progressão. Perguntas ambíguas sem subintenção confiável usam fallback contextual. `TOKEN_NAO_RELACIONADO` nunca é redirecionado por palavras isoladas da pergunta.

A interface mantém uma engine por sessão em `st.session_state`. Ações que alteram o puzzle salvam o resultado e executam `st.rerun()` controladamente; o resultado continua disponível no rerun e a ação não é repetida. O classificador permanece sob `st.cache_resource`, enquanto o estado do puzzle continua isolado por sessão. A animação de vitória usa uma flag de sessão, aparece uma vez por partida e é restaurada no reinício.

Para executar a aplicação:

```powershell
python -m streamlit run streamlit_app.py
```

Para executar testes sem o modelo real e a integração separadamente:

```powershell
python -m pytest -m "not integracao"
python -m pytest -m integracao
```

A avaliação funcional independente das métricas da Semana 2 está documentada em [`docs/avaliacao_funcional_dicas.md`](docs/avaliacao_funcional_dicas.md), com casos em `data/avaliacao_funcional_dicas.csv`. Os arquivos originais `data/frases_referencia.csv`, `data/teste_manual.csv` e os resultados experimentais não são alterados por essa avaliação.

O relatório técnico das correções, da validação e das limitações da Semana 4 está em [`docs/correcoes_semana4.md`](docs/correcoes_semana4.md). Ele diferencia resultados executados com mocks, testes locais, smoke do Streamlit e integração com o modelo real.

As evidências visuais incluem `01-inicial.png`, `02-relogio.png`, `03-cofre-fechado.png`, `04-cofre-aberto.png`, `05-fotografia.png`, `06-mural-completo.png`, `07-enciclopedia.png`, `08-simbolos.png`, `09-dica.png`, `10-vitoria.png` e `11-reset.png`.
