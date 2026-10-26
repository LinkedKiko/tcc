# Análise dos erros do melhor classificador — Semana 5

Fonte: `results/embeddings_threshold_teste_erros.csv`. São 14 erros em 60 perguntas, preservados sem alteração.

As categorias abaixo são uma triagem interpretável baseada nos tokens e em termos explícitos da pergunta. Não substituem a revisão humana de cada rótulo.

## Distribuição por categoria

| categoria | quantidade |
| --- | --- |
| confusão entre objetos relacionados | 4 |
| formulação vaga ou indireta | 3 |
| aceitação indevida | 3 |
| sobreposição semântica entre pistas | 2 |
| rejeição indevida | 2 |

## Distribuição por token esperado

| token_esperado | quantidade |
| --- | --- |
| TOKEN_COFRE | 2 |
| TOKEN_LIVRO | 2 |
| TOKEN_MURAL | 2 |
| TOKEN_NAO_RELACIONADO | 3 |
| TOKEN_RELOGIO | 3 |
| TOKEN_SIMBOLOS | 2 |

## Erros observados

| pergunta | token_esperado | token_previsto | similaridade | categoria_erro | base_categorizacao |
| --- | --- | --- | --- | --- | --- |
| Os rabiscos gravados perto da fechadura eletrônica apontam uma ordem? | TOKEN_SIMBOLOS | TOKEN_COFRE | 0.6606742493690875 | confusão entre objetos relacionados | triagem por rótulos e termos; requer revisão humana |
| O que exatamente devo decifrar para conseguir liberar a saída? | TOKEN_SIMBOLOS | TOKEN_COFRE | 0.5474056096828398 | formulação vaga ou indireta | triagem por rótulos e termos; requer revisão humana |
| O livro de animais serve para descobrir o nome de cada símbolo? | TOKEN_LIVRO | TOKEN_MURAL | 0.8010953127728373 | sobreposição semântica entre pistas | triagem por rótulos e termos; requer revisão humana |
| A obra ilustrada pode ensinar a correspondência entre criaturas e símbolos? | TOKEN_LIVRO | TOKEN_MURAL | 0.7305284673207549 | sobreposição semântica entre pistas | triagem por rótulos e termos; requer revisão humana |
| As figuras repetidas no desenho podem virar algarismos? | TOKEN_MURAL | TOKEN_SIMBOLOS | 0.7056383201255205 | confusão entre objetos relacionados | triagem por rótulos e termos; requer revisão humana |
| Como eu transformo o que está retratado em números? | TOKEN_MURAL | TOKEN_RELOGIO | 0.6152156505448847 | formulação vaga ou indireta | triagem por rótulos e termos; requer revisão humana |
| Vale a pena destrancar o cofre antes de mexer na porta? | TOKEN_COFRE | TOKEN_NAO_RELACIONADO | 0.4713383902952075 | rejeição indevida | triagem por rótulos e termos; requer revisão humana |
| Preciso de um código para destrancar aquele móvel pequeno? | TOKEN_COFRE | TOKEN_NAO_RELACIONADO | 0.4687797480771095 | rejeição indevida | triagem por rótulos e termos; requer revisão humana |
| Os ponteiros parados fornecem os números da abertura? | TOKEN_RELOGIO | TOKEN_COFRE | 0.6787058320391561 | confusão entre objetos relacionados | triagem por rótulos e termos; requer revisão humana |
| O instante em que tudo paralisou tem relação com a abertura? | TOKEN_RELOGIO | TOKEN_COFRE | 0.6145149007998432 | formulação vaga ou indireta | triagem por rótulos e termos; requer revisão humana |
| Como uso aquilo que está travado para descobrir a combinação? | TOKEN_RELOGIO | TOKEN_COFRE | 0.6925273262062931 | confusão entre objetos relacionados | triagem por rótulos e termos; requer revisão humana |
| Que livro de ficção você recomenda? | TOKEN_NAO_RELACIONADO | TOKEN_LIVRO | 0.5543342879052173 | aceitação indevida | triagem por rótulos e termos; requer revisão humana |
| Como criar uma senha realmente segura para o banco? | TOKEN_NAO_RELACIONADO | TOKEN_COFRE | 0.5897736259687586 | aceitação indevida | triagem por rótulos e termos; requer revisão humana |
| Quantas espécies de aves existem no Brasil? | TOKEN_NAO_RELACIONADO | TOKEN_MURAL | 0.5861310184977268 | aceitação indevida | triagem por rótulos e termos; requer revisão humana |

## Leitura por token

Os erros envolvendo `TOKEN_COFRE`, `TOKEN_RELOGIO`, `TOKEN_MURAL`, `TOKEN_LIVRO` e `TOKEN_NAO_RELACIONADO` foram mantidos separados no CSV para permitir distinguir rejeições indevidas, aceitações indevidas e confusões entre intenções do puzzle. A matriz de confusão original permanece em `results/embeddings_threshold_teste_matriz_confusao.csv` e as métricas por token em `results/embeddings_threshold_teste_por_classe.csv`.

A categoria `rejeição indevida` corresponde a pergunta legítima abaixo do limiar; `aceitação indevida` corresponde a entrada não relacionada aceita; os demais casos são confusões entre intenções relacionadas. Nenhuma frase de referência foi alterada.
