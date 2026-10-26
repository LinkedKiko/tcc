# Comparação dos classificadores

Conjunto avaliado: **teste** (60 perguntas, idênticas para todos os métodos).

## Métricas agregadas

| metodo                   | conjunto | threshold | total | acertos | erros | acuracia | macro_precision | macro_recall | macro_f1 |
| ------------------------ | -------- | --------- | ----- | ------- | ----- | -------- | --------------- | ------------ | -------- |
| Palavras-chave           | teste    | n/d       | 60    | 34      | 26    | 0.5667   | 0.6619          | 0.5630       | 0.5904   |
| Embeddings sem threshold | teste    | n/d       | 60    | 40      | 20    | 0.6667   | 0.7150          | 0.7037       | 0.6712   |

## Leitura dos resultados

- Maior acurácia: **Embeddings sem threshold** (66.67%).
- Maior Macro F1: **Embeddings sem threshold** (0.6712).

## Dificuldade por token (F1 por classe)

| token                 | f1_Palavras-chave | f1_Embeddings sem threshold | suporte |
| --------------------- | ----------------- | --------------------------- | ------- |
| TOKEN_SIMBOLOS        | 0.4211            | 0.7368                      | 9       |
| TOKEN_LIVRO           | 0.8235            | 0.8235                      | 9       |
| TOKEN_MURAL           | 0.7500            | 0.7000                      | 9       |
| TOKEN_COFRE           | 0.6250            | 0.6667                      | 9       |
| TOKEN_RELOGIO         | 0.4615            | 0.6000                      | 9       |
| TOKEN_NAO_RELACIONADO | 0.4615            | 0.5000                      | 15      |

## Dificuldade por nível das perguntas

| metodo                   | dificuldade | total | acertos | acuracia |
| ------------------------ | ----------- | ----- | ------- | -------- |
| Palavras-chave           | dificil     | 21    | 3       | 0.1429   |
| Palavras-chave           | facil       | 24    | 24      | 1.0000   |
| Palavras-chave           | media       | 15    | 7       | 0.4667   |
| Embeddings sem threshold | dificil     | 21    | 11      | 0.5238   |
| Embeddings sem threshold | facil       | 24    | 17      | 0.7083   |
| Embeddings sem threshold | media       | 15    | 12      | 0.8000   |

## Divergências pergunta a pergunta

### Embeddings acertaram e palavras-chave erraram (16 pergunta(s))

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

### Palavras-chave acertaram e embeddings erraram (10 pergunta(s))

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

### Ambos acertaram (24 pergunta(s))

- "Esses desenhos misteriosos ajudam a escolher os números do painel?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_SIMBOLOS` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.6798
- "Devo anotar a ordem exata dos símbolos antes de digitar no painel?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_SIMBOLOS` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.7135
- "Esses símbolos ao lado da entrada correspondem a dígitos?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_SIMBOLOS` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.8263
- "A marcação próxima da porta pode indicar uma sequência numérica?"  
  esperado=`TOKEN_SIMBOLOS` | keywords=`TOKEN_SIMBOLOS` | embeddings=`TOKEN_SIMBOLOS` | similaridade=0.6828
- "Que informação da enciclopédia pode esclarecer a linguagem da sala?"  
  esperado=`TOKEN_LIVRO` | keywords=`TOKEN_LIVRO` | embeddings=`TOKEN_LIVRO` | similaridade=0.6324
- _(+19 pergunta(s) nesta categoria)_

### Ambos erraram (10 pergunta(s))

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
