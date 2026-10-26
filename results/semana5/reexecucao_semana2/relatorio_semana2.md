# Relatório da Semana 2

Documento gerado automaticamente por `evaluate.py`. Todos os números vêm da execução real dos classificadores sobre os arquivos em `data/`.

- Gerado em: 2026-10-25 21:01:04

## 1. Dataset

Divisão estratificada determinística com `RANDOM_STATE = 42` e 40% das frases para validação.

Perguntas por token e por partição:

| token_esperado        | completo | validacao | teste |
| --------------------- | -------- | --------- | ----- |
| TOKEN_SIMBOLOS        | 15       | 6         | 9     |
| TOKEN_LIVRO           | 15       | 6         | 9     |
| TOKEN_MURAL           | 15       | 6         | 9     |
| TOKEN_COFRE           | 15       | 6         | 9     |
| TOKEN_RELOGIO         | 15       | 6         | 9     |
| TOKEN_NAO_RELACIONADO | 25       | 10        | 15    |
| TOTAL                 | 100      | 40        | 60    |

Distribuição por nível de dificuldade no conjunto completo:

| token_esperado        | dificil | facil | media |
| --------------------- | ------- | ----- | ----- |
| TOKEN_COFRE           | 5       | 5     | 5     |
| TOKEN_LIVRO           | 5       | 5     | 5     |
| TOKEN_MURAL           | 5       | 5     | 5     |
| TOKEN_NAO_RELACIONADO | 6       | 14    | 5     |
| TOKEN_RELOGIO         | 5       | 5     | 5     |
| TOKEN_SIMBOLOS        | 5       | 5     | 5     |

Verificações de integridade:

- Frases de referência: 90
- Frases de avaliação: 100
- Frases idênticas entre referência e avaliação: 0
- Frases presentes em validação e teste ao mesmo tempo: 0

## 2. Classificador por palavras-chave

| metodo         | conjunto | threshold | total | acertos | erros | acuracia | macro_precision | macro_recall | macro_f1 |
| -------------- | -------- | --------- | ----- | ------- | ----- | -------- | --------------- | ------------ | -------- |
| Palavras-chave | teste    | n/d       | 60    | 34      | 26    | 0.5667   | 0.6619          | 0.5630       | 0.5904   |

Métricas por token:

| token                 | precisao | recall | f1     | suporte |
| --------------------- | -------- | ------ | ------ | ------- |
| TOKEN_SIMBOLOS        | 0.4000   | 0.4444 | 0.4211 | 9       |
| TOKEN_LIVRO           | 0.8750   | 0.7778 | 0.8235 | 9       |
| TOKEN_MURAL           | 0.8571   | 0.6667 | 0.7500 | 9       |
| TOKEN_COFRE           | 0.7143   | 0.5556 | 0.6250 | 9       |
| TOKEN_RELOGIO         | 0.7500   | 0.3333 | 0.4615 | 9       |
| TOKEN_NAO_RELACIONADO | 0.3750   | 0.6000 | 0.4615 | 15      |

Matriz de confusão (linhas = token esperado, colunas = token previsto):

| classe                     | previsto_TOKEN_SIMBOLOS | previsto_TOKEN_LIVRO | previsto_TOKEN_MURAL | previsto_TOKEN_COFRE | previsto_TOKEN_RELOGIO | previsto_TOKEN_NAO_RELACIONADO |
| -------------------------- | ----------------------- | -------------------- | -------------------- | -------------------- | ---------------------- | ------------------------------ |
| real_TOKEN_SIMBOLOS        | 4                       | 0                    | 0                    | 0                    | 0                      | 5                              |
| real_TOKEN_LIVRO           | 2                       | 7                    | 0                    | 0                    | 0                      | 0                              |
| real_TOKEN_MURAL           | 0                       | 0                    | 6                    | 0                    | 0                      | 3                              |
| real_TOKEN_COFRE           | 1                       | 0                    | 0                    | 5                    | 0                      | 3                              |
| real_TOKEN_RELOGIO         | 1                       | 0                    | 0                    | 1                    | 3                      | 4                              |
| real_TOKEN_NAO_RELACIONADO | 2                       | 1                    | 1                    | 1                    | 1                      | 9                              |

Acurácia por nível de dificuldade:

| metodo         | dificuldade | total | acertos | acuracia |
| -------------- | ----------- | ----- | ------- | -------- |
| Palavras-chave | dificil     | 21    | 3       | 0.1429   |
| Palavras-chave | facil       | 24    | 24      | 1.0000   |
| Palavras-chave | media       | 15    | 7       | 0.4667   |

## 3. Classificador por embeddings (sem threshold)

| metodo                   | conjunto | threshold | total | acertos | erros | acuracia | macro_precision | macro_recall | macro_f1 |
| ------------------------ | -------- | --------- | ----- | ------- | ----- | -------- | --------------- | ------------ | -------- |
| Embeddings sem threshold | teste    | n/d       | 60    | 40      | 20    | 0.6667   | 0.7150          | 0.7037       | 0.6712   |

Métricas por token:

| token                 | precisao | recall | f1     | suporte |
| --------------------- | -------- | ------ | ------ | ------- |
| TOKEN_SIMBOLOS        | 0.7000   | 0.7778 | 0.7368 | 9       |
| TOKEN_LIVRO           | 0.8750   | 0.7778 | 0.8235 | 9       |
| TOKEN_MURAL           | 0.6364   | 0.7778 | 0.7000 | 9       |
| TOKEN_COFRE           | 0.5333   | 0.8889 | 0.6667 | 9       |
| TOKEN_RELOGIO         | 0.5455   | 0.6667 | 0.6000 | 9       |
| TOKEN_NAO_RELACIONADO | 1.0000   | 0.3333 | 0.5000 | 15      |

Matriz de confusão (linhas = token esperado, colunas = token previsto):

| classe                     | previsto_TOKEN_SIMBOLOS | previsto_TOKEN_LIVRO | previsto_TOKEN_MURAL | previsto_TOKEN_COFRE | previsto_TOKEN_RELOGIO | previsto_TOKEN_NAO_RELACIONADO |
| -------------------------- | ----------------------- | -------------------- | -------------------- | -------------------- | ---------------------- | ------------------------------ |
| real_TOKEN_SIMBOLOS        | 7                       | 0                    | 0                    | 2                    | 0                      | 0                              |
| real_TOKEN_LIVRO           | 0                       | 7                    | 2                    | 0                    | 0                      | 0                              |
| real_TOKEN_MURAL           | 1                       | 0                    | 7                    | 0                    | 1                      | 0                              |
| real_TOKEN_COFRE           | 1                       | 0                    | 0                    | 8                    | 0                      | 0                              |
| real_TOKEN_RELOGIO         | 0                       | 0                    | 0                    | 3                    | 6                      | 0                              |
| real_TOKEN_NAO_RELACIONADO | 1                       | 1                    | 2                    | 2                    | 4                      | 5                              |

Acurácia por nível de dificuldade:

| metodo                   | dificuldade | total | acertos | acuracia |
| ------------------------ | ----------- | ----- | ------- | -------- |
| Embeddings sem threshold | dificil     | 21    | 11      | 0.5238   |
| Embeddings sem threshold | facil       | 24    | 17      | 0.7083   |
| Embeddings sem threshold | media       | 15    | 12      | 0.8000   |

## 4. Análise de thresholds

Resultados obtidos **exclusivamente no conjunto de validação**. O conjunto de teste não participou da escolha do limiar.

| threshold | acuracia | macro_f1 | aceitas_corretamente | rejeitadas_corretamente | falsas_aceitacoes | falsas_rejeicoes | classificadas_corretamente | taxa_rejeicao_correta | taxa_rejeicao_incorreta |
| --------- | -------- | -------- | -------------------- | ----------------------- | ----------------- | ---------------- | -------------------------- | --------------------- | ----------------------- |
| 0.4000    | 0.7500   | 0.7409   | 30                   | 8                       | 2                 | 0                | 22                         | 0.8000                | 0.0000                  |
| 0.4500    | 0.7500   | 0.7409   | 30                   | 8                       | 2                 | 0                | 22                         | 0.8000                | 0.0000                  |
| 0.5000    | 0.7750   | 0.7585   | 29                   | 9                       | 1                 | 1                | 22                         | 0.9000                | 0.0333                  |
| 0.5500    | 0.7750   | 0.7474   | 27                   | 10                      | 0                 | 3                | 21                         | 1.0000                | 0.1000                  |
| 0.6000    | 0.6250   | 0.6158   | 20                   | 10                      | 0                 | 10               | 15                         | 1.0000                | 0.3333                  |
| 0.6500    | 0.5500   | 0.5397   | 15                   | 10                      | 0                 | 15               | 12                         | 1.0000                | 0.5000                  |
| 0.7000    | 0.4500   | 0.4101   | 10                   | 10                      | 0                 | 20               | 8                          | 1.0000                | 0.6667                  |
| 0.7500    | 0.4250   | 0.3750   | 8                    | 10                      | 0                 | 22               | 7                          | 1.0000                | 0.7333                  |
| 0.8000    | 0.3000   | 0.1543   | 3                    | 10                      | 0                 | 27               | 2                          | 1.0000                | 0.9000                  |

- Critério de seleção: maior Macro F1; empate resolvido pela maior acuracia e, persistindo, pelo menor threshold
- Threshold selecionado: **0.50**
- Macro F1 na validação: 0.7585
- Acurácia na validação: 0.7750
- Limiares empatados no critério: 1

Similaridade máxima frente às cinco intenções do puzzle:

| grupo            | count | mean   | min    | max    |
| ---------------- | ----- | ------ | ------ | ------ |
| nao_relacionadas | 15    | 0.3833 | 0.1562 | 0.5898 |
| relacionadas     | 45    | 0.6715 | 0.4688 | 0.8522 |

## 5. Avaliação com o threshold selecionado

| metodo                        | conjunto | threshold | total | acertos | erros | acuracia | macro_precision | macro_recall | macro_f1 |
| ----------------------------- | -------- | --------- | ----- | ------- | ----- | -------- | --------------- | ------------ | -------- |
| Embeddings com threshold 0.50 | teste    | 0.5000    | 60    | 46      | 14    | 0.7667   | 0.7838          | 0.7630       | 0.7663   |

Métricas por token:

| token                 | precisao | recall | f1     | suporte |
| --------------------- | -------- | ------ | ------ | ------- |
| TOKEN_SIMBOLOS        | 0.8750   | 0.7778 | 0.8235 | 9       |
| TOKEN_LIVRO           | 0.8750   | 0.7778 | 0.8235 | 9       |
| TOKEN_MURAL           | 0.7000   | 0.7778 | 0.7368 | 9       |
| TOKEN_COFRE           | 0.5385   | 0.7778 | 0.6364 | 9       |
| TOKEN_RELOGIO         | 0.8571   | 0.6667 | 0.7500 | 9       |
| TOKEN_NAO_RELACIONADO | 0.8571   | 0.8000 | 0.8276 | 15      |

Matriz de confusão (linhas = token esperado, colunas = token previsto):

| classe                     | previsto_TOKEN_SIMBOLOS | previsto_TOKEN_LIVRO | previsto_TOKEN_MURAL | previsto_TOKEN_COFRE | previsto_TOKEN_RELOGIO | previsto_TOKEN_NAO_RELACIONADO |
| -------------------------- | ----------------------- | -------------------- | -------------------- | -------------------- | ---------------------- | ------------------------------ |
| real_TOKEN_SIMBOLOS        | 7                       | 0                    | 0                    | 2                    | 0                      | 0                              |
| real_TOKEN_LIVRO           | 0                       | 7                    | 2                    | 0                    | 0                      | 0                              |
| real_TOKEN_MURAL           | 1                       | 0                    | 7                    | 0                    | 1                      | 0                              |
| real_TOKEN_COFRE           | 0                       | 0                    | 0                    | 7                    | 0                      | 2                              |
| real_TOKEN_RELOGIO         | 0                       | 0                    | 0                    | 3                    | 6                      | 0                              |
| real_TOKEN_NAO_RELACIONADO | 0                       | 1                    | 1                    | 1                    | 0                      | 12                             |

Acurácia por nível de dificuldade:

| metodo                        | dificuldade | total | acertos | acuracia |
| ----------------------------- | ----------- | ----- | ------- | -------- |
| Embeddings com threshold 0.50 | dificil     | 21    | 14      | 0.6667   |
| Embeddings com threshold 0.50 | facil       | 24    | 20      | 0.8333   |
| Embeddings com threshold 0.50 | media       | 15    | 12      | 0.8000   |

## 6. Comparação dos cenários

| metodo                        | conjunto | threshold | total | acertos | erros | acuracia | macro_precision | macro_recall | macro_f1 |
| ----------------------------- | -------- | --------- | ----- | ------- | ----- | -------- | --------------- | ------------ | -------- |
| Palavras-chave                | teste    | n/d       | 60    | 34      | 26    | 0.5667   | 0.6619          | 0.5630       | 0.5904   |
| Embeddings sem threshold      | teste    | n/d       | 60    | 40      | 20    | 0.6667   | 0.7150          | 0.7037       | 0.6712   |
| Embeddings com threshold 0.50 | teste    | 0.5000    | 60    | 46      | 14    | 0.7667   | 0.7838          | 0.7630       | 0.7663   |

## 7. Análise de erros

### Embeddings acertaram e palavras-chave erraram

Total: 16 pergunta(s).

- "Preciso decifrar aquela anotação antiga para liberar a passagem?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.5379
- "Aquela inscrição estranha na superfície vertical serve de orientação?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.5962
- "A ordem em que as marcas aparecem altera o resultado?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.6208
- "Onde encontro uma referência escrita que explique essa linguagem?"  
  esperado=`TOKEN_LIVRO` | keywords=`TOKEN_SIMBOLOS` | embeddings=`TOKEN_LIVRO` | similaridade=0.6662
- "Como eu descubro a correspondência entre os nomes dos animais e aquela escrita?"  
  esperado=`TOKEN_LIVRO` | keywords=`TOKEN_SIMBOLOS` | embeddings=`TOKEN_LIVRO` | similaridade=0.7645
- _(+11 pergunta(s) nesta categoria)_

### Palavras-chave acertaram e embeddings erraram

Total: 10 pergunta(s).

- "O livro de animais serve para descobrir o nome de cada símbolo?"  
  esperado=`TOKEN_LIVRO` | keywords=`TOKEN_LIVRO` | embeddings=`TOKEN_MURAL` | similaridade=0.8011
- "A obra ilustrada pode ensinar a correspondência entre criaturas e símbolos?"  
  esperado=`TOKEN_LIVRO` | keywords=`TOKEN_LIVRO` | embeddings=`TOKEN_MURAL` | similaridade=0.7305
- "As figuras repetidas no desenho podem virar algarismos?"  
  esperado=`TOKEN_MURAL` | keywords=`TOKEN_MURAL` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.7056
- "Vale a pena destrancar o cofre antes de mexer na porta?"  
  esperado=`TOKEN_COFRE` | keywords=`TOKEN_COFRE` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.4713
- "Os ponteiros parados fornecem os números da abertura?"  
  esperado=`TOKEN_RELOGIO` | keywords=`TOKEN_RELOGIO` | embeddings=`TOKEN_COFRE` | similaridade=0.6787
- _(+5 pergunta(s) nesta categoria)_

### Ambos acertaram

Total: 24 pergunta(s).

### Ambos erraram

Total: 10 pergunta(s).

- "Os rabiscos gravados perto da fechadura eletrônica apontam uma ordem?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_COFRE` | similaridade=0.6607
- "O que exatamente devo decifrar para conseguir liberar a saída?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_COFRE` | similaridade=0.5474
- "Como eu transformo o que está retratado em números?"  
  esperado=`TOKEN_MURAL` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_RELOGIO` | similaridade=0.6152
- "O instante em que tudo paralisou tem relação com a abertura?"  
  esperado=`TOKEN_RELOGIO` | keywords=`TOKEN_NAO_RELACIONADO` | embeddings=`TOKEN_COFRE` | similaridade=0.6145
- "Onde fica guardada a documentação do meu veículo?"  
  esperado=`TOKEN_NAO_RELACIONADO` | keywords=`TOKEN_COFRE` | embeddings=`TOKEN_COFRE` | similaridade=0.4520
- _(+5 pergunta(s) nesta categoria)_

### Perguntas não relacionadas aceitas incorretamente (modo de rejeição)

Total: 3 pergunta(s).

- "Que livro de ficção você recomenda?"  
  esperado=`TOKEN_NAO_RELACIONADO` | previsto=`TOKEN_LIVRO` | similaridade=0.5543
- "Como criar uma senha realmente segura para o banco?"  
  esperado=`TOKEN_NAO_RELACIONADO` | previsto=`TOKEN_COFRE` | similaridade=0.5898
- "Quantas espécies de aves existem no Brasil?"  
  esperado=`TOKEN_NAO_RELACIONADO` | previsto=`TOKEN_MURAL` | similaridade=0.5861

### Perguntas relacionadas rejeitadas incorretamente (modo de rejeição)

Total: 2 pergunta(s).

- "Vale a pena destrancar o cofre antes de mexer na porta?"  
  esperado=`TOKEN_COFRE` | previsto=`TOKEN_NAO_RELACIONADO` | similaridade=0.4713
- "Preciso de um código para destrancar aquele móvel pequeno?"  
  esperado=`TOKEN_COFRE` | previsto=`TOKEN_NAO_RELACIONADO` | similaridade=0.4688

## 8. Limitações

As observações abaixo delimitam o alcance dos números apresentados.

- **Dimensão reduzida do dataset.** São 90 frases de referência e 100 perguntas
  de avaliação. Com 6 perguntas por classe na validação e 9 no teste (25 e 15
  para `TOKEN_NAO_RELACIONADO`), cada acerto ou erro desloca as métricas em
  vários pontos percentuais. Os intervalos de confiança são largos e as
  diferenças pequenas entre métodos não devem ser lidas como conclusivas.
- **Seleção manual das perguntas.** As perguntas de avaliação e seus rótulos
  foram elaborados pelo próprio autor, a partir da descrição do puzzle. Não
  houve anotação independente nem medição de concordância entre avaliadores,
  portanto os rótulos devem ser tratados como uma proposta sujeita a revisão
  humana, não como padrão-ouro validado.
- **Possível ambiguidade semântica.** Algumas intenções do puzzle se sobrepõem
  por construção: o relógio abre o cofre, o cofre completa o mural e o mural dá
  valor aos símbolos. Perguntas que citam dois elementos admitem mais de uma
  leitura plausível, e o rótulo escolhido reflete a intenção principal julgada
  pelo autor.
- **Influência das frases de referência.** O classificador semântico só
  consegue reconhecer uma intenção se houver uma frase de referência próxima.
  O desempenho medido é tanto uma avaliação do modelo de embeddings quanto do
  conjunto de 90 frases escolhido na Semana 1.
- **Sensibilidade do threshold.** O limiar foi escolhido em uma grade
  exploratória de nove valores, usando apenas 40 perguntas de validação. Um
  conjunto de validação diferente poderia indicar outro valor, e os limiares
  observados são específicos deste modelo de embeddings.
- **Ausência de avaliação com jogadores reais.** Nenhuma pergunta foi coletada
  de pessoas jogando o puzzle. A formulação espontânea de dúvidas durante o
  jogo pode diferir das perguntas escritas em ambiente controlado.
- **Baseline definido a priori.** As listas de palavras-chave foram montadas a
  partir da descrição do puzzle e do vocabulário das frases de referência, sem
  consultar o conjunto de avaliação. Um baseline ajustado iterativamente aos
  dados teria desempenho diferente.


## 9. Arquivos gerados

- `results/semana5/reexecucao_semana2/analise_thresholds.csv` (analise_thresholds)
- `results/semana5/reexecucao_semana2/baseline_teste_detalhado.csv` (baseline_teste_detalhado)
- `results/semana5/reexecucao_semana2/baseline_teste_erros.csv` (baseline_teste_erros)
- `results/semana5/reexecucao_semana2/baseline_teste_matriz_confusao.csv` (baseline_teste_matriz_confusao)
- `results/semana5/reexecucao_semana2/baseline_teste_metricas.csv` (baseline_teste_metricas)
- `results/semana5/reexecucao_semana2/baseline_teste_por_classe.csv` (baseline_teste_por_classe)
- `results/semana5/reexecucao_semana2/comparacao_classificadores.csv` (comparacao_por_pergunta)
- `results/semana5/reexecucao_semana2/comparacao_metricas.csv` (comparacao_metricas)
- `results/semana5/reexecucao_semana2/comparacao_resumo.md` (comparacao_resumo)
- `results/semana5/reexecucao_semana2/distribuicao_similaridades_validacao.csv` (distribuicao_similaridades)
- `results/semana5/reexecucao_semana2/embeddings_teste_detalhado.csv` (embeddings_teste_detalhado)
- `results/semana5/reexecucao_semana2/embeddings_teste_erros.csv` (embeddings_teste_erros)
- `results/semana5/reexecucao_semana2/embeddings_teste_matriz_confusao.csv` (embeddings_teste_matriz_confusao)
- `results/semana5/reexecucao_semana2/embeddings_teste_metricas.csv` (embeddings_teste_metricas)
- `results/semana5/reexecucao_semana2/embeddings_teste_por_classe.csv` (embeddings_teste_por_classe)
- `results/semana5/reexecucao_semana2/embeddings_threshold_teste_detalhado.csv` (embeddings_threshold_teste_detalhado)
- `results/semana5/reexecucao_semana2/embeddings_threshold_teste_erros.csv` (embeddings_threshold_teste_erros)
- `results/semana5/reexecucao_semana2/embeddings_threshold_teste_matriz_confusao.csv` (embeddings_threshold_teste_matriz_confusao)
- `results/semana5/reexecucao_semana2/embeddings_threshold_teste_metricas.csv` (embeddings_threshold_teste_metricas)
- `results/semana5/reexecucao_semana2/embeddings_threshold_teste_por_classe.csv` (embeddings_threshold_teste_por_classe)
- `results/semana5/reexecucao_semana2/particao_resumo.csv` (particao_resumo)
- `results/semana5/reexecucao_semana2/particao_teste.csv` (particao_teste)
- `results/semana5/reexecucao_semana2/particao_validacao.csv` (particao_validacao)
- `results/semana5/reexecucao_semana2/threshold_selecionado.csv` (threshold_selecionado)
- `results/semana5/reexecucao_semana2/threshold_selecionado_teste.csv` (threshold_avaliado_no_teste)
