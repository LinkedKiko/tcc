# Resultados e discussão — Semana 5

## 1. Ambiente e condições experimentais

A avaliação utilizou o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, com embeddings de sentenças e similaridade de cosseno. O ambiente do benchmark final foi Windows 11, Python 3.14.7, execução em CPU, 12 processadores lógicos, Sentence Transformers 6.1.0, PyTorch 2.14.1+cpu, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2, Streamlit 1.66.1 e pytest 9.1.1. A identificação completa está em `results/semana5/benchmark_final/execucao_benchmark_20261026_011428_065614_4b5e8d03/ambiente_experimental.json`.

A configuração histórica foi preservada: `RANDOM_STATE = 42`, validação de 40%, teste de 60% e threshold selecionado de 0,50. O benchmark final usou uma amostra independente e estratificada de 30 perguntas, cinco por token. Não houve fine-tuning, ajuste sobre o teste ou alteração das frases de referência.

## 2. Caracterização dos dados

O conjunto de referências contém 90 frases, distribuídas em seis tokens, com 15 frases por token. O conjunto de avaliação contém 100 perguntas: 15 para cada uma das cinco intenções do puzzle e 25 para `TOKEN_NAO_RELACIONADO`. A divisão estratificada reproduzida resultou em 40 perguntas de validação e 60 de teste: seis exemplos por intenção do puzzle e dez não relacionados na validação; nove por intenção do puzzle e 15 não relacionados no teste.

A verificação automática não encontrou valores ausentes essenciais, duplicatas normalizadas, frases idênticas entre referência e avaliação ou sobreposição entre validação e teste. Os hashes SHA-256 estão em `results/semana5/integridade_dados.json`. Foram encontrados candidatos a ambiguidade em perguntas que mencionam simultaneamente objetos conectados, por exemplo relógio/cofre, cofre/porta, enciclopédia/animais/símbolos e cofre/mural. Esses itens são alertas para revisão humana e não tiveram rótulos modificados.

## 3. Baseline por palavras-chave

O baseline lexical obteve 34 acertos em 60 perguntas, acurácia de 56,67% e Macro F1 de 0,5904. O método normaliza o texto e conta correspondências de palavras-chave, com desempate determinístico. Seu desempenho é limitado quando a pergunta usa formulações indiretas ou sinônimos que não aparecem no vocabulário lexical previamente definido.

## 4. Classificador por embeddings sem threshold

O modo padrão obteve 40 acertos em 60 perguntas, acurácia de 66,67%, Macro Precision de 0,7150, Macro Recall de 0,7037 e Macro F1 de 0,6712. Nesse modo, cada pergunta é comparada com as 90 referências, incluindo as referências de `TOKEN_NAO_RELACIONADO`, e a maior similaridade determina a decisão.

## 5. Classificador com rejeição

O cenário histórico com threshold obteve 46 acertos em 60 perguntas, acurácia de 76,67%, Macro Precision de 0,7838, Macro Recall de 0,7630 e Macro F1 de 0,7663. Nesse modo, a comparação é feita apenas com as 75 referências das cinco intenções do puzzle. Quando a maior similaridade fica abaixo de 0,50, a pergunta recebe `TOKEN_NAO_RELACIONADO`.

A diferença entre os cenários não consiste apenas em acrescentar um threshold: a composição das referências e a política de decisão também mudam. Consequentemente, o ganho observado não deve ser atribuído exclusivamente ao limiar.

## 6. Comparação entre configurações

| Método | Acertos | Total | Acurácia | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Palavras-chave | 34 | 60 | 56,67% | 0,6619 | 0,5630 | 0,5904 |
| Embeddings sem threshold | 40 | 60 | 66,67% | 0,7150 | 0,7037 | 0,6712 |
| Embeddings com rejeição 0,50 | 46 | 60 | 76,67% | 0,7838 | 0,7630 | 0,7663 |

Os valores coincidem com `results/comparacao_metricas.csv`. A tabela detalhada, as políticas e as fontes estão em `results/semana5/comparacao_final.csv` e `.md`.

## 7. Análise do threshold

A grade de thresholds foi avaliada exclusivamente na validação. O critério de seleção foi maior Macro F1, seguido de maior acurácia e, em empate persistente, menor threshold. O valor selecionado foi 0,50, com Macro F1 de 0,7585 e acurácia de 0,7750 na validação. O teste somente recebeu a configuração já selecionada.

O threshold melhorou a rejeição de entradas externas, mas não elimina a sobreposição semântica. No teste, houve duas rejeições indevidas de perguntas legítimas e três aceitações indevidas de perguntas não relacionadas.

## 8. Resultados por intenção e matrizes de confusão

As métricas por token estão em `results/embeddings_threshold_teste_por_classe.csv` e a matriz de confusão em `results/embeddings_threshold_teste_matriz_confusao.csv`. Os melhores F1 foram observados para `TOKEN_NAO_RELACIONADO` (0,8276), `TOKEN_SIMBOLOS` e `TOKEN_LIVRO` (0,8235). O menor F1 foi o de `TOKEN_COFRE` (0,6364), seguido de `TOKEN_MURAL` (0,7368) e `TOKEN_RELOGIO` (0,7500).

As confusões mais recorrentes envolveram relógio/cofre, mural/símbolos e livro/mural. A matriz confirma que os objetos relacionados pela própria progressão do puzzle são semanticamente próximos.

## 9. Análise de erros

O melhor classificador apresentou 14 erros. O arquivo `results/semana5/analise_erros.csv` mantém pergunta, tokens, similaridade e categoria de triagem. A análise diferenciou rejeição indevida, aceitação indevida, confusão entre objetos relacionados, sobreposição semântica e formulação vaga ou indireta.

As categorias são uma triagem transparente baseada nos rótulos e em termos da pergunta, não uma anotação humana definitiva. Não foram feitas alterações em referências, rótulos ou parâmetros a partir desses erros.

## 10. Tempo de processamento

A medição final foi executada com o MiniLM real em CPU, no identificador
`benchmark_20261026_011428_065614_4b5e8d03`. A amostra contém 30 perguntas,
cinco por cada um dos seis tokens, selecionadas com seed 42. Foram usados 30
repetições e três aquecimentos por configuração. A primeira inferência ficou
separada das 900 observações subsequentes (`30 × 30`) de cada classificador.

| Medição | Valor |
|---|---:|
| Inicialização do modelo | 6603,499500 ms |
| Preparação das referências | 259,379700 ms |
| Primeira inferência — palavras-chave | 0,036800 ms |
| Primeira inferência — embeddings padrão | 28,134600 ms |
| Primeira inferência — embeddings rejeição 0,50 | 14,868200 ms |
| Subsequente — palavras-chave (média / mediana / DP / mín. / máx. / P95) | 0,016296 / 0,016800 / 0,004006 / 0,008000 / 0,059200 / 0,021315 ms |
| Subsequente — embeddings padrão (média / mediana / DP / mín. / máx. / P95) | 19,514245 / 17,313150 / 8,932556 / 12,970100 / 162,402100 / 29,333215 ms |
| Subsequente — embeddings rejeição 0,50 (média / mediana / DP / mín. / máx. / P95) | 15,701110 / 15,372400 / 2,454191 / 13,243300 / 54,616900 / 17,682300 ms |

Os valores foram lidos de `latencia_resumo.csv`; a inicialização e a
preparação vêm de `ambiente_experimental.json`, e o relatório Markdown foi
gerado pelos mesmos artefatos. A gravação dos arquivos ficou fora dos
intervalos de `time.perf_counter()`. A execução preliminar antiga, com 15
perguntas do prefixo de uma única classe, permanece preservada e não foi
misturada ao benchmark final. Esses tempos são observacionais e dependem do
hardware, cache, versões e estado do processo.

## 11. Testes funcionais e integração

A execução final foi registrada em `results/semana5/testes_finais/` com XML,
logs, JSON e relatório Markdown. A suíte completa coletou **213 testes**, com
213 aprovados, 0 falhas, 0 ignorados e 0 erros de coleta, código de saída 0 e
12,698 s no XML do pytest (15,710 s medidos pelo registrador). Os números de
execuções direcionadas não foram somados à suíte completa:

| Grupo | Coletados | Aprovados | Falhas | Ignorados | Erros | Código |
|---|---:|---:|---:|---:|---:|---:|
| Suíte completa | 213 | 213 | 0 | 0 | 0 | 0 |
| Streamlit | 14 | 14 | 0 | 0 | 0 | 0 |
| Integração com MiniLM real | 3 | 3 | 0 | 0 | 0 | 0 |
| Testes específicos da Semana 5 | 11 | 11 | 0 | 0 | 0 | 0 |

O grupo de integração carregou efetivamente o modelo obrigatório e verificou
dimensão, similaridade válida e concordância entre lote e chamadas individuais.
Os logs anteriores, incluindo a falha transitória causada pela ausência de
`tabulate` no relatório intermediário, permanecem em
`testes_finais/historico_execucoes/`; a execução final não depende dessa
biblioteca opcional.
## 12. Adequação das dicas

A avaliação funcional contém 27 cenários referentes a 15 perguntas distintas. Por pergunta distinta, considerando uma pergunta correta apenas quando todas as suas repetições de estado foram corretas, foram obtidas 10/15 (66,67%). Por cenário, foram obtidos 20/27 tokens corretos (74,07%).

A adequação humana registrada para as dicas foi 20/27 (74,07%). O sucesso completo, definido pela conjunção de token correto e dica adequada, foi 18/27 (66,67%). Todas as 27 dicas respeitaram o progresso registrado e não apresentaram spoiler segundo os julgamentos existentes. Os quatro quadrantes foram: token errado/dica inadequada 5; token errado/dica adequada 2; token correto/dica inadequada 2; token correto/dica adequada 18.

Os julgamentos dos casos AV014 e AV020 foram preservados como `assunto_adequado = Sim` e `dica_adequada = Sim` no CSV atual. O relatório registra a tensão metodológica entre utilidade da dica e pertinência do token, sem substituir a avaliação humana.

## 13. Discussão da hipótese

No conjunto estudado, os dois cenários baseados em embeddings obtiveram métricas superiores ao baseline lexical, tanto sem rejeição quanto na configuração histórica com rejeição. Esse resultado é compatível com a hipótese de que a representação semântica ajuda a identificar intenções em perguntas formuladas com vocabulário indireto.

A evidência não comprova superioridade universal. A comparação envolve protocolos diferentes entre modo padrão e modo de rejeição, um único modelo, um cenário específico e um conjunto pequeno de perguntas. A conclusão é restrita aos dados, referências e condições experimentais documentados.

## 14. Limitações e ameaças à validade

- Conjunto de teste reduzido, com 60 perguntas.
- Número limitado de intenções e dependência de um único puzzle.
- Perguntas elaboradas para um cenário específico.
- Sobreposição semântica entre objetos conectados pela progressão.
- Dependência da qualidade e cobertura das frases de referência.
- Uso de um único modelo de embeddings.
- Protocolos diferentes entre modo padrão e modo de rejeição.
- Uso de um único threshold selecionado na validação.
- Avaliação funcional exploratória, com julgamentos humanos.
- Repetição das mesmas perguntas em estados diferentes.
- Possível subjetividade na adequação das dicas.
- Ausência de testes com grande número de jogadores.
- Dependência das características específicas do puzzle.
- A validação automatizada cobre a interface Streamlit e o motor; a execução manual do servidor não faz parte da contagem automatizada.

## 15. Rastreabilidade

Os artefatos finais estão separados por execução:

- Benchmark: `results/semana5/benchmark_final/execucao_benchmark_20261026_011428_065614_4b5e8d03/`.
- Reprodução: `results/semana5/reproducao_final/`, com `manifesto_execucao.json`, `status_execucao.json`, logs e os CSVs derivados.
- Testes: `results/semana5/testes_finais/`, com XML, logs e `resumo_testes.json`.
- Ambiente instalável: `requirements.txt` e `requirements-lock.txt`.
- Histórico anterior: `results/semana5/reexecucao_semana2/` e os artefatos preliminares de latência, preservados sem mistura.

A tag e o commit finais devem ser conferidos em `results/semana5/versao_final.json` e no Git; o arquivo só pode registrar identificadores realmente existentes.