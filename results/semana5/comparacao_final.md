# Comparação final dos classificadores — Semana 5

Os três cenários abaixo são os registros históricos da Semana 2. Nenhum PNG ou CSV histórico foi sobrescrito.

## Tabela consolidada

| metodo | conjunto | threshold | total | acertos | erros | acuracia | macro_precision | macro_recall | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Palavras-chave | teste |  | 60 | 34 | 26 | 0.5666666666666667 | 0.6619047619047619 | 0.562962962962963 | 0.5904431610700961 |
| Embeddings sem threshold | teste |  | 60 | 40 | 20 | 0.6666666666666666 | 0.7150252525252525 | 0.7037037037037037 | 0.671173030615755 |
| Embeddings com threshold 0.50 | teste | 0.5 | 60 | 46 | 14 | 0.7666666666666667 | 0.7837912087912088 | 0.762962962962963 | 0.766308462008793 |

## Protocolos

- **Palavras-chave**: classes `TOKEN_SIMBOLOS, TOKEN_LIVRO, TOKEN_MURAL, TOKEN_COFRE, TOKEN_RELOGIO, TOKEN_NAO_RELACIONADO`; TOKEN_NAO_RELACIONADO quando nenhuma palavra-chave foi encontrada; empate determinístico. Fonte: `results/comparacao_metricas.csv`
- **Embeddings sem threshold**: classes `TOKEN_SIMBOLOS, TOKEN_LIVRO, TOKEN_MURAL, TOKEN_COFRE, TOKEN_RELOGIO, TOKEN_NAO_RELACIONADO`; Modo padrão: comparação com as 90 referências, inclusive TOKEN_NAO_RELACIONADO; sem threshold. Fonte: `results/comparacao_metricas.csv; results/embeddings_teste_detalhado.csv`
- **Embeddings com threshold 0.50**: classes `TOKEN_SIMBOLOS, TOKEN_LIVRO, TOKEN_MURAL, TOKEN_COFRE, TOKEN_RELOGIO, TOKEN_NAO_RELACIONADO`; Modo rejeição: comparação somente com as 75 referências das cinco intenções; abaixo de 0,50 retorna TOKEN_NAO_RELACIONADO. Fonte: `results/comparacao_metricas.csv; results/embeddings_threshold_teste_detalhado.csv; results/threshold_selecionado.csv`

## Conferência contra os valores históricos esperados

- `Palavras-chave`: **coincidente**.
- `Embeddings sem threshold`: **coincidente**.
- `Embeddings com threshold 0.50`: **coincidente**.

## Interpretação metodológica

O cenário sem threshold e o cenário com threshold não são a mesma configuração com um único parâmetro alterado. O modo padrão usa as 90 referências, incluindo a classe não relacionada, enquanto o modo de rejeição usa somente 75 referências das cinco intenções e aplica o limiar de 0,50. Portanto, o ganho observado não é atribuído exclusivamente ao threshold.

O threshold foi selecionado na validação pela maior Macro F1, com desempates por acurácia e menor threshold, e somente depois aplicado ao teste. Este relatório não reajusta parâmetros.
