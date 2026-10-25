# Metodologia — Semana 2

Material de apoio para a redação do TCC **"Aplicação de embeddings semânticos
na geração de tokens de decisão para esclarecimento de dúvidas em jogos de
puzzle"**. O documento descreve o que foi implementado, como os experimentos
são conduzidos e quais são os limites do que pode ser afirmado a partir deles.

Os números de cada execução ficam em `results/relatorio_semana2.md`, gerado
automaticamente por `evaluate.py`. Este documento descreve o método; o
relatório descreve os resultados.

---

## 1. Construção do dataset

### 1.1 Frases de referência

| Item | Valor |
|---|---|
| Arquivo | `data/frases_referencia.csv` |
| Colunas | `frase`, `token` |
| Total de frases | 90 |
| Tokens | 6 |
| Frases por token | 15 |

As 90 frases foram definidas na Semana 1 e **não foram alteradas**. Elas não são
dados de treino no sentido estatístico: nenhum parâmetro é ajustado a partir
delas. Funcionam como protótipos de intenção, isto é, como o conjunto de
vetores contra os quais toda pergunta nova é comparada.

Consequência metodológica importante: o desempenho medido avalia
simultaneamente o modelo de embeddings e a qualidade da escolha dessas 90
frases. Uma intenção cuja formulação típica não esteja representada entre as 15
frases correspondentes tende a ser mal classificada, mesmo que o modelo de
embeddings seja adequado.

### 1.2 Perguntas de avaliação

| Item | Valor |
|---|---|
| Arquivo | `data/teste_manual.csv` |
| Colunas | `frase`, `token_esperado`, `dificuldade` |
| Total de perguntas | 100 |

Distribuição por intenção:

| Token esperado | Quantidade |
|---|---:|
| `TOKEN_SIMBOLOS` | 15 |
| `TOKEN_LIVRO` | 15 |
| `TOKEN_MURAL` | 15 |
| `TOKEN_COFRE` | 15 |
| `TOKEN_RELOGIO` | 15 |
| `TOKEN_NAO_RELACIONADO` | 25 |
| **Total** | **100** |

`TOKEN_NAO_RELACIONADO` recebeu mais exemplos porque é a classe mais
heterogênea: perguntas irrelevantes não formam um grupo semanticamente
coerente, e um único assunto fora do jogo não representa a variedade do que um
jogador pode perguntar.

### 1.3 Critérios de elaboração das perguntas

As perguntas de avaliação foram escritas a partir da descrição do puzzle,
seguindo estes critérios:

1. **Independência lexical.** Nenhuma frase de avaliação reproduz literalmente
   uma frase de referência. A verificação é automática: `evaluate.py` compara
   os dois conjuntos e registra quantas frases coincidem (o esperado é zero).
2. **Intenção principal identificável.** Cada pergunta admite uma leitura
   dominante. Formulações que seriam ambíguas até para um avaliador humano
   foram evitadas.
3. **Diversidade de formulação.** Uso de sinônimos ("caixa trancada" para
   cofre, "aparelho de medir o tempo" para relógio), construções indiretas e
   perguntas que não citam o objeto pelo nome.
4. **Plausibilidade.** As perguntas imitam dúvidas que um jogador formularia
   diante de cada elemento da sala.

### 1.4 Níveis de dificuldade

A coluna `dificuldade` classifica a distância lexical entre a pergunta e o
vocabulário das frases de referência. Ela não mede dificuldade de resolução do
puzzle, e sim o quanto o reconhecimento da intenção depende de semântica em vez
de casamento de palavras.

| Nível | Critério | Exemplo |
|---|---|---|
| `facil` | Menciona explicitamente o objeto principal | "Preciso contar quantas libélulas estão no mural?" |
| `media` | Usa sinônimos ou referências indiretas | "Aquele aparelho de medir o tempo paralisado é uma dica?" |
| `dificil` | Expressa a intenção sem as palavras centrais das referências | "Aquilo que marca o tempo na sala guarda três valores úteis?" |

As cinco intenções do puzzle têm 5 perguntas de cada nível (5 + 5 + 5 = 15).

Para `TOKEN_NAO_RELACIONADO` o eixo é diferente: mede a sobreposição lexical
com o vocabulário do jogo, não a distância até uma intenção.

| Nível | Critério | Quantidade |
|---|---|---:|
| `facil` | Assunto claramente externo, sem vocabulário em comum | 14 |
| `media` | Contém termos de uso geral que também aparecem no jogo | 5 |
| `dificil` | Contém um termo central do jogo com intenção externa | 6 |

As 6 perguntas `dificil` são **negativos difíceis** deliberados: trazem uma
palavra do vocabulário do puzzle ("livro", "horário", "senha", "sequência",
"escrita", "espécies") aplicada a um assunto externo. Exemplo: "Que horário
costuma ter menos trânsito na cidade?" contém "horário", mas não é uma pergunta
sobre o relógio da sala. Elas existem para distinguir dois comportamentos que a
rejeição por limiar precisa separar: reconhecer vocabulário e reconhecer
intenção. Sua presença torna as métricas de `TOKEN_NAO_RELACIONADO` mais
exigentes do que um conjunto composto apenas de assuntos distantes.

### 1.5 Limitações do dataset elaborado manualmente

- Perguntas e rótulos foram produzidos pelo mesmo autor que implementou o
  sistema. Não houve anotação independente nem medição de concordância entre
  avaliadores (por exemplo, kappa de Cohen). **Os rótulos são uma proposta
  inicial de avaliação, sujeita a revisão humana, e não um padrão-ouro
  validado por especialistas.**
- Com 15 perguntas por intenção do puzzle, a partição de teste fica com 9 por
  classe. Um único erro altera o recall daquela classe em cerca de 11 pontos
  percentuais.
- Perguntas escritas fora do contexto de jogo podem diferir sistematicamente
  das dúvidas formuladas por alguém efetivamente jogando.
- As intenções do puzzle são encadeadas por construção (relógio → cofre →
  mural → símbolos, com o livro como tradutor). Perguntas que citam dois
  elementos têm rótulo definido pela intenção julgada principal, o que
  introduz uma decisão subjetiva.

---

## 2. Separação entre validação e teste

A divisão evita que o sistema seja ajustado aos mesmos exemplos usados para
relatar o desempenho.

| Partição | Fração | Perguntas | Função |
|---|---:|---:|---|
| Validação | 40% | 40 | investigar limiares, comparar configurações, inspecionar problemas |
| Teste | 60% | 60 | medir o desempenho final, já com a configuração fixada |

Procedimento, implementado em `src/dataset.py`:

1. `carregar_avaliacao` lê e valida o CSV: colunas obrigatórias, frases vazias,
   tokens desconhecidos e frases duplicadas.
2. `dividir_validacao_teste` usa `sklearn.model_selection.train_test_split` com
   `stratify=token_esperado`, `train_size=0.40` e `random_state=42`
   (`RANDOM_STATE` em `src/config.py`).
3. A estratificação preserva a proporção das classes: 6 perguntas por intenção
   do puzzle e 10 não relacionadas na validação; 9 e 15 no teste.
4. A função verifica explicitamente que nenhuma frase aparece nas duas
   partições e falha se isso ocorrer.
5. As partições são gravadas em `results/particao_validacao.csv` e
   `results/particao_teste.csv`. O arquivo `data/teste_manual.csv` permanece a
   fonte original e não é modificado.

Como a semente é fixa e a ordenação é restaurada pelo índice original, a
divisão é determinística: execuções repetidas produzem exatamente as mesmas
partições.

Nenhuma frase de avaliação é adicionada ao conjunto de referência do
classificador. Validação e teste nunca influenciam os vetores de comparação.

---

## 3. Método baseado em embeddings

### 3.1 Modelo

```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Um Sentence-BERT multilíngue, treinado para aproximar paráfrases no espaço
vetorial. Produz vetores de 384 dimensões e cobre português. A escolha se
justifica por três restrições do trabalho: execução local, sem custo de API e
sem treinamento adicional.

O modelo é carregado uma única vez, na construção de `PuzzleClassifier`
(`src/classifier.py`), e reutilizado durante toda a execução.

### 3.2 Geração dos vetores

Na inicialização, as 90 frases de referência são codificadas em uma única
chamada ao modelo, resultando em uma matriz `90 × 384` mantida em memória.

Para cada pergunta nova, apenas o vetor da pergunta é calculado. Os embeddings
de referência nunca são recalculados. Nas avaliações, as perguntas de uma
partição são codificadas em lote e a matriz de similaridades é reaproveitada
por todos os experimentos daquela execução, inclusive pelos nove limiares da
grade.

### 3.3 Similaridade de cosseno

A comparação usa a similaridade de cosseno entre o vetor da pergunta e cada
vetor de referência:

$$
\operatorname{sim}(A,B) = \frac{A \cdot B}{\lVert A \rVert \, \lVert B \rVert}
$$

O numerador é o produto interno e o denominador é o produto das normas
euclidianas. O resultado varia em $[-1, 1]$: 1 indica vetores com a mesma
direção, 0 indica ortogonalidade e -1 indica direções opostas. Como a medida
depende apenas do ângulo, perguntas longas e curtas sobre o mesmo assunto podem
obter similaridade alta.

Na prática, embeddings de frases curtas em português produzidos por este modelo
raramente assumem valores negativos, e a faixa observada fica próxima de
$[0, 1]$. O cálculo usa `sklearn.metrics.pairwise.cosine_similarity` e os
valores são limitados a $[-1, 1]$ para absorver erro numérico.

### 3.4 Seleção da referência e geração do token

1. Calcular a similaridade da pergunta com as 90 referências.
2. Ordenar em ordem decrescente. Empates são resolvidos pelo menor índice do
   CSV, o que torna o ranking reprodutível.
3. Tomar a referência de maior similaridade.
4. Emitir o token associado a essa referência.
5. Expor as três referências mais próximas (`TOP_K = 3`) com seus tokens e
   similaridades, permitindo inspecionar por que uma decisão foi tomada.
6. `src/responses.py` mapeia o token para a dica apresentada ao jogador.

O token é a unidade de decisão: o sistema não gera texto livre, apenas escolhe
qual dica pré-escrita apresentar. É o que permite que a dica seja controlada
pelo projetista do jogo.

**O maior valor de similaridade não garante que a classificação esteja
correta.** A decisão é relativa: a referência vencedora é apenas a mais próxima
entre as disponíveis. Se nenhuma referência for realmente adequada, a
comparação ainda devolve um vencedor, com similaridade possivelmente alta por
proximidade de vocabulário e não de intenção. Essa é precisamente a motivação
da análise de limiares da Seção 5.

---

## 4. Baseline por palavras-chave

Implementado em `src/keyword_classifier.py`, sem usar embeddings. Serve para
estabelecer quanto do desempenho é obtido por casamento lexical simples, o que
dá sentido à comparação: sem baseline, não é possível afirmar que o ganho
observado venha da representação semântica.

### 4.1 Listas de palavras-chave

Cada um dos cinco tokens do puzzle recebeu uma lista de termos. O vocabulário
foi definido **a priori**, a partir de duas fontes: a descrição textual do
puzzle e o vocabulário já presente em `data/frases_referencia.csv` (o
conhecimento declarado do sistema).

As perguntas de `data/teste_manual.csv` não foram consultadas para montar nem
ajustar as listas. As listas também não foram revisadas depois de observar os
resultados do conjunto de teste. Essa restrição é necessária para que a
comparação seja honesta: um baseline ajustado iterativamente aos dados de
avaliação não seria uma linha de base, e sim um método otimizado sobre o
conjunto em que é medido.

### 4.2 Palavras genéricas excluídas

Os termos `senha`, `codigo`, `combinacao`, `numero`, `animal`, `pista`,
`puzzle`, `sala`, `abrir` e `tempo` foram **deliberadamente excluídos** de
todas as listas. Cada um aparece em frases de referência de mais de uma
intenção, ou em perguntas não relacionadas. Exemplos:

- `senha` aparece nas referências de `TOKEN_SIMBOLOS`, `TOKEN_COFRE` e
  `TOKEN_RELOGIO`, e também em uma pergunta não relacionada sobre a senha do
  computador;
- `animal` aparece nas referências de `TOKEN_LIVRO` e `TOKEN_MURAL`;
- `tempo` aparece em contextos do relógio e em perguntas sobre clima.

Incluí-los produziria atribuições arbitrárias, cuja origem seria a ambiguidade
do termo e não a regra de decisão.

### 4.3 Normalização textual

1. conversão para minúsculas (`str.casefold`);
2. remoção de acentos por decomposição Unicode NFKD, descartando os sinais
   diacríticos ("relógio" → "relogio");
3. substituição de pontuação e símbolos por espaço;
4. separação em palavras por espaços em branco.

A comparação é feita por **palavra completa**: "relogio" casa com "relogio",
mas não com "relojoeiro" nem com "relogioeiro". Evita correspondências
acidentais por subcadeias. Expressões com mais de uma palavra, quando usadas,
são buscadas como sequência contígua.

### 4.4 Contagem e regra de decisão

1. Para cada token, contar quantas **palavras-chave distintas** aparecem na
   pergunta. A repetição de um mesmo termo vale um ponto, de modo que repetir
   uma palavra não domina a decisão.
2. Selecionar o token com a maior contagem.
3. Se nenhuma palavra-chave for encontrada (contagem máxima igual a zero),
   devolver `TOKEN_NAO_RELACIONADO`.

### 4.5 Regra de desempate

Entre os tokens com a maior contagem, vence o primeiro desta ordem fixa:

```
TOKEN_RELOGIO → TOKEN_COFRE → TOKEN_LIVRO → TOKEN_MURAL → TOKEN_SIMBOLOS
```

A ordem foi definida a priori pela cadeia de dependências do puzzle, percorrida
do elemento mais específico para o mais genérico: o relógio abre o cofre, o
cofre completa o mural, o mural dá valor aos símbolos, e o livro traduz a
linguagem. Os termos de `TOKEN_SIMBOLOS` ("parede", "sequencia", "escrita",
"porta") são os mais compartilhados com as demais intenções nas frases de
referência, por isso ficam em último lugar.

A regra é determinística: a mesma pergunta produz sempre o mesmo token,
independentemente da ordem de iteração dos dicionários. Tokens eventualmente
adicionados sem posição explícita são colocados no fim, em ordem alfabética.

O resultado do baseline expõe `similaridade = None`. O método não calcula
similaridade de cosseno, e **nenhum valor é inventado em seu lugar**: nos CSVs,
o campo permanece vazio.

### 4.6 Limitações previsíveis do baseline

Por construção, o baseline não reconhece sinônimos ("caixa trancada"),
paráfrases ("aparelho de medir o tempo") nem intenções expressas sem o
vocabulário listado. Também é sensível a termos do jogo usados fora de
contexto. Essas limitações são esperadas e fazem parte do que a comparação se
propõe a medir; elas não são o resultado do experimento.

---

## 5. Limiar de similaridade (threshold)

### 5.1 Finalidade

Na implementação da Semana 1, o classificador sempre devolve a referência de
maior similaridade. Diante de uma pergunta sem relação com o jogo, ele ainda
aponta uma das intenções. Para um sistema de dicas, isso significa entregar uma
dica irrelevante em vez de informar que a pergunta está fora do escopo.

O limiar introduz a possibilidade de **abstenção**: se nada é suficientemente
próximo, o sistema não escolhe.

### 5.2 Duas abordagens comparadas

O trabalho compara duas formas conceitualmente distintas de tratar perguntas
irrelevantes.

**Abordagem A — classe no dataset de referência** (`MODO_PADRAO`)

`TOKEN_NAO_RELACIONADO` é uma classe como as outras, com 15 frases de
referência. A pergunta é comparada com todas as 90 referências e vence a maior
similaridade. É o comportamento da Semana 1 e é o que ocorre quando
`SIMILARITY_THRESHOLD = None`.

Pressuposto implícito: perguntas irrelevantes formam um grupo coeso no espaço
de embeddings. Esse pressuposto é questionável, porque "fora do escopo" não é
um assunto, mas o complemento de um conjunto de assuntos. Quinze frases sobre
capitais, receitas e futebol não cobrem o universo do que é externo ao jogo.

**Abordagem B — rejeição por limiar** (`MODO_REJEICAO`)

Apenas as 75 referências das cinco intenções do puzzle participam da escolha.
`TOKEN_NAO_RELACIONADO` **não** compete: ele deixa de ser uma classe e passa a
ser o resultado de uma decisão de abstenção.

```
Pergunta
    ↓
Embedding da pergunta
    ↓
Comparação com as referências das cinco intenções do puzzle
    ↓
Maior similaridade encontrada
    ↓
Verificação do limiar
    ↓
similaridade >= threshold  →  token da intenção mais próxima
similaridade <  threshold  →  TOKEN_NAO_RELACIONADO
```

Pressuposto: perguntas do jogo ficam mais próximas de alguma referência do que
perguntas externas. Testar esse pressuposto é o objetivo do gráfico de
distribuição das similaridades (`results/grafico_distribuicao_similaridades.png`):
se as duas distribuições se sobrepõem fortemente, nenhum limiar separa bem os
dois grupos, e isso é um resultado relevante para o TCC.

### 5.3 Classificação e rejeição são decisões diferentes

A análise registra as duas dimensões separadamente:

| Métrica | Pergunta esperada | Decisão |
|---|---|---|
| `aceitas_corretamente` | relacionada | aceita (independe do token) |
| `classificadas_corretamente` | relacionada | aceita **e** com o token correto |
| `falsas_rejeicoes` | relacionada | rejeitada |
| `rejeitadas_corretamente` | não relacionada | rejeitada |
| `falsas_aceitacoes` | não relacionada | aceita |

Uma pergunta relacionada que é aceita com o token errado conta como aceitação
correta, mas **não** como classificação correta. Sem essa distinção, um limiar
muito permissivo pareceria bom por aceitar tudo, mesmo errando a intenção.

Taxas derivadas:

- `taxa_rejeicao_correta` = `rejeitadas_corretamente` / total de não relacionadas;
- `taxa_rejeicao_incorreta` = `falsas_rejeicoes` / total de relacionadas;
- `acuracia_intencoes_aceitas` = `classificadas_corretamente` / `aceitas_corretamente`.

### 5.4 Grade exploratória

```python
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
```

São nove valores em passos de 0,05. A grade é **exploratória**: não representa
limites teoricamente corretos para modelos de embedding em geral. Escalas de
similaridade variam entre modelos, e um valor adequado aqui não se transfere
para outro modelo. A lista pode ser estendida pela linha de comando
(`--thresholds`).

### 5.5 Critério de escolha

A grade é avaliada **exclusivamente no conjunto de validação**. O critério,
determinístico e documentado:

1. maior Macro F1;
2. em caso de empate, maior acurácia;
3. persistindo o empate, o menor limiar.

O terceiro nível favorece a configuração menos restritiva: rejeitar uma dúvida
legítima do jogador prejudica mais a experiência do que aceitar uma pergunta
fora de contexto, que no pior caso recebe uma dica inútil.

O limiar escolhido é registrado em `results/threshold_selecionado.csv` e,
somente depois, aplicado **uma única vez** ao conjunto de teste, com o
resultado em `results/threshold_selecionado_teste.csv`. **O conjunto de teste
não participa da escolha do limiar.**

### 5.6 Limitações da abordagem por limiar

- O limiar é escolhido a partir de 40 perguntas de validação. Outro conjunto de
  validação poderia indicar outro valor.
- A grade de nove pontos pode não conter o melhor valor; o ótimo pode estar
  entre dois pontos avaliados.
- O valor é específico deste modelo de embeddings e deste conjunto de
  referência. Trocar qualquer um dos dois exige repetir a análise.
- Um limiar único trata todas as intenções da mesma forma. Nada garante que a
  escala de similaridade seja homogênea entre as cinco intenções; limiares por
  classe são uma extensão possível, não investigada nesta semana.
- Existe um compromisso inerente: elevar o limiar aumenta a rejeição correta de
  perguntas externas e, ao mesmo tempo, a rejeição incorreta de perguntas
  legítimas. Não existe escolha sem custo, apenas um ponto de equilíbrio.

---

## 6. Métricas de avaliação

Calculadas com `scikit-learn` em `src/evaluation.py`. Nas métricas por classe,
divisões por zero recebem valor 0 (`zero_division=0`), situação que ocorre
quando uma classe não aparece entre as predições de uma partição.

Para uma classe $c$, define-se:

- $VP$ (verdadeiros positivos): perguntas de $c$ previstas como $c$;
- $FP$ (falsos positivos): perguntas de outras classes previstas como $c$;
- $FN$ (falsos negativos): perguntas de $c$ previstas como outra classe.

### 6.1 Acurácia

$$
\text{Acurácia} = \frac{\text{classificações corretas}}{\text{total de perguntas}}
$$

Simples de interpretar, mas sensível ao desbalanceamento. Com 25 das 100
perguntas em `TOKEN_NAO_RELACIONADO`, um classificador que devolvesse sempre
esse token já alcançaria 25% de acurácia sem compreender nada. Por isso a
acurácia nunca é relatada sozinha.

### 6.2 Precisão

$$
\text{Precisão}_c = \frac{VP_c}{VP_c + FP_c}
$$

Entre as perguntas às quais o sistema atribuiu o token $c$, quantas realmente
eram de $c$. Precisão baixa em uma intenção significa que o sistema entrega a
dica daquele elemento para perguntas que tratavam de outro.

### 6.3 Recall

$$
\text{Recall}_c = \frac{VP_c}{VP_c + FN_c}
$$

Entre as perguntas que realmente eram de $c$, quantas foram identificadas.
Recall baixo significa que dúvidas sobre aquele elemento passam sem ser
reconhecidas.

### 6.4 F1-score

$$
F1_c = 2 \cdot \frac{\text{Precisão}_c \cdot \text{Recall}_c}
{\text{Precisão}_c + \text{Recall}_c}
$$

Média harmônica das duas. A média harmônica penaliza desequilíbrio: um método
com precisão 1,0 e recall 0,1 obtém F1 de aproximadamente 0,18, e não 0,55.

### 6.5 Médias macro

$$
\text{Macro } F1 = \frac{1}{|C|}\sum_{c \in C} F1_c
$$

A média aritmética simples sobre as seis classes dá o mesmo peso a todas,
independentemente de quantas perguntas cada uma possui. É a métrica principal
deste trabalho, porque interessa que o sistema reconheça as cinco intenções do
puzzle, e não apenas que acerte a classe mais numerosa. Macro Precision e Macro
Recall seguem a mesma definição.

### 6.6 Matriz de confusão

Matriz $6 \times 6$ em que a linha indica o token esperado e a coluna o token
previsto. A diagonal contém os acertos; cada célula fora da diagonal mostra
**qual** confusão ocorreu, informação que as métricas agregadas não revelam.
Confusões concentradas entre intenções encadeadas do puzzle (relógio e cofre,
mural e livro) têm interpretação diferente de erros distribuídos
aleatoriamente.

Versões em CSV e em PNG são geradas para cada cenário avaliado.

### 6.7 Registro dos erros

Toda avaliação grava a lista completa de perguntas classificadas
incorretamente, com pergunta, token esperado, token previsto, similaridade
obtida (vazia no baseline) e nível de dificuldade. A análise de divergências
separa a comparação entre métodos em quatro casos: embeddings corretos e
palavras-chave incorretas; palavras-chave corretas e embeddings incorretos;
ambos corretos; ambos incorretos. Categorias vazias são declaradas vazias, sem
exemplos fabricados.

---

## 7. Procedimento experimental

Sequência executada por `python evaluate.py --modo completo`:

1. carregar `data/teste_manual.csv` e validar o formato;
2. produzir a divisão determinística em validação (40) e teste (60) e gravá-la
   em `results/`;
3. conferir a integridade: frases em comum com o conjunto de referência e
   sobreposição entre partições (ambas devem ser zero);
4. carregar o modelo de embeddings e codificar as 90 frases de referência;
5. codificar as perguntas de cada partição em lote, uma única vez;
6. avaliar os nove limiares **na validação** e selecionar um pelo critério da
   Seção 5.5;
7. avaliar, **no conjunto de teste**, os três cenários: baseline por
   palavras-chave, embeddings sem limiar e embeddings com o limiar selecionado;
8. comparar os métodos pergunta a pergunta e classificar as divergências;
9. gerar os gráficos e as matrizes de confusão;
10. escrever `results/relatorio_semana2.md`.

O modelo é carregado no máximo uma vez por execução e nenhuma matriz de
similaridade é recalculada.

---

## 8. Resultados

Os resultados completos de cada execução ficam em
`results/relatorio_semana2.md`, com as tabelas por classe, as matrizes de
confusão e os exemplos de erro produzidos pela rodada.

### 8.1 Resultados da execução de referência

Valores obtidos pela execução real de `python evaluate.py --modo completo` em
12/10/2026, com `RANDOM_STATE = 42`. São reproduzíveis: a divisão é
determinística e nenhuma etapa usa sorteio adicional.

Seleção do limiar, **no conjunto de validação** (40 perguntas):

| Threshold | Acurácia | Macro F1 | Falsas aceitações | Falsas rejeições |
|---:|---:|---:|---:|---:|
| 0,40 | 0,7500 | 0,7409 | 2 | 0 |
| 0,45 | 0,7500 | 0,7409 | 2 | 0 |
| **0,50** | **0,7750** | **0,7585** | **1** | **1** |
| 0,55 | 0,7750 | 0,7474 | 0 | 3 |
| 0,60 | 0,6250 | 0,6158 | 0 | 10 |
| 0,65 | 0,5500 | 0,5397 | 0 | 15 |
| 0,70 | 0,4500 | 0,4101 | 0 | 20 |
| 0,75 | 0,4250 | 0,3750 | 0 | 22 |
| 0,80 | 0,3000 | 0,1543 | 0 | 27 |

Limiar selecionado: **0,50**, sem empate no critério.

Desempenho **no conjunto de teste** (60 perguntas, idênticas para os três
cenários), avaliado uma única vez após a escolha do limiar:

| Cenário | Acertos | Acurácia | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Palavras-chave | 34/60 | 56,67% | 0,6619 | 0,5630 | 0,5904 |
| Embeddings sem limiar | 40/60 | 66,67% | 0,7150 | 0,7037 | 0,6712 |
| Embeddings com limiar 0,50 | 46/60 | 76,67% | 0,7838 | 0,7630 | 0,7663 |

Divergências pergunta a pergunta entre os dois métodos (sem limiar): ambos
corretos em 24 perguntas, ambos incorretos em 10, embeddings corretos e
palavras-chave incorretas em 16, palavras-chave corretas e embeddings
incorretos em 10. **As duas direções de divergência existem**: o método
semântico não domina o baseline caso a caso, apenas no agregado.

### 8.2 Observações sobre os resultados

Registradas como leitura dos dados, não como conclusões definitivas; o
tamanho do conjunto não sustenta afirmações fortes.

- A separação entre perguntas relacionadas e não relacionadas, medida pela
  similaridade máxima frente às cinco intenções, foi ampla na validação:
  relacionadas com média 0,668 (mínimo 0,497) e não relacionadas com média
  0,326 (máximo 0,508). As distribuições quase não se sobrepõem, o que explica
  por que a rejeição por limiar funcionou melhor que a classe de referência.
- O ganho da Abordagem B sobre a Abordagem A concentrou-se em
  `TOKEN_NAO_RELACIONADO`: o recall dessa classe subiu de 0,3333 para 0,8000.
  A hipótese da Seção 5.2 — de que 15 frases de referência não cobrem o
  universo do que é externo ao jogo — é consistente com esse resultado.
- O baseline por palavras-chave acertou 100% das perguntas `facil` (24/24) e
  14,29% das `dificil` (3/21). O método semântico foi mais estável entre os
  níveis. Essa é a diferença que a coluna `dificuldade` foi criada para medir.
- Os erros residuais do método semântico concentram-se entre intenções
  encadeadas do puzzle: três perguntas de `TOKEN_RELOGIO` foram atribuídas a
  `TOKEN_COFRE` e duas de `TOKEN_LIVRO` a `TOKEN_MURAL`. São confusões entre
  elementos que o próprio puzzle relaciona, não erros arbitrários.
- As três perguntas não relacionadas aceitas indevidamente pelo limiar são
  exatamente negativos difíceis ("Que livro de ficção você recomenda?", "Como
  criar uma senha realmente segura para o banco?", "Quantas espécies de aves
  existem no Brasil?"), o que confirma que o limiar responde a vocabulário e
  não apenas a intenção.

Os arquivos gerados são:

| Arquivo | Conteúdo |
|---|---|
| `results/relatorio_semana2.md` | relatório completo da execução |
| `results/comparacao_resumo.md` | leitura da comparação entre métodos |
| `results/particao_validacao.csv`, `results/particao_teste.csv` | composição das partições |
| `results/analise_thresholds.csv` | grade de limiares na validação |
| `results/threshold_selecionado.csv` | limiar escolhido e critério |
| `results/threshold_selecionado_teste.csv` | avaliação única no teste |
| `results/comparacao_classificadores.csv` | predições dos dois métodos pergunta a pergunta |
| `results/*_detalhado.csv`, `results/*_erros.csv` | predições e erros por método |
| `results/*_metricas.csv`, `results/*_por_classe.csv` | métricas agregadas e por classe |
| `results/grafico_*.png`, `results/matriz_confusao_*.png` | gráficos e matrizes |

Ao citar números no texto do TCC, usar os valores desses arquivos. **Nenhum
resultado deve ser estimado, arredondado de memória ou reproduzido de execuções
anteriores sem conferência.** Se alguma análise não puder ser executada no
ambiente disponível, a seção correspondente do relatório é marcada como não
executada.

---

## 9. O que não foi feito nesta semana

Registrado para delimitar o escopo:

- nenhum treinamento ou ajuste fino de modelo;
- nenhum LLM generativo participa da classificação;
- nenhuma API externa paga;
- nenhum banco de dados ou banco vetorial: a busca é exaustiva sobre 90 vetores,
  o que é adequado nesta escala;
- nenhuma interface gráfica;
- nenhuma comparação entre modelos de embeddings diferentes;
- nenhuma coleta de perguntas com jogadores reais;
- nenhum teste de significância estatística entre os métodos. Com 60 perguntas
  no conjunto de teste, diferenças pequenas de acurácia não são distinguíveis
  de variação amostral, e o relatório não afirma superioridade a partir delas.
