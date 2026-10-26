# Relatório de latência — Semana 5

**Identificador da execução:** `benchmark_20261026_011428_065614_4b5e8d03`.
**Status:** `executado`.

Este relatório foi gerado a partir do `ambiente_experimental.json`, `latencia_execucoes.csv` e `latencia_resumo.csv` desta mesma execução. Os tempos usam `time.perf_counter()` e não incluem escrita em disco.

## Amostra estratificada

- Perguntas: **30**; por classe: **5**.
- Semente: **42**.
- Classes: `TOKEN_COFRE, TOKEN_LIVRO, TOKEN_MURAL, TOKEN_NAO_RELACIONADO, TOKEN_RELOGIO, TOKEN_SIMBOLOS`.
- Distribuição: `{'TOKEN_COFRE': 5, 'TOKEN_LIVRO': 5, 'TOKEN_MURAL': 5, 'TOKEN_NAO_RELACIONADO': 5, 'TOKEN_RELOGIO': 5, 'TOKEN_SIMBOLOS': 5}`.
- A amostra completa está em `amostra_perguntas.csv`; a medição preserva a mesma ordem para os três classificadores.

## Ambiente

- **data_execucao_utc:** '2026-10-26T01:14:28.088703+00:00'
- **sistema_operacional:** 'Windows-11-10.0.26200-SP0'
- **python:** '3.14.7'
- **processador:** 'AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD'
- **cpu_logicos:** 12
- **memoria_gb:** None
- **dispositivo:** 'cpu'
- **sentence_transformers:** '6.1.0'
- **torch:** '2.14.1+cpu'
- **numpy:** '2.5.3'
- **pandas:** '3.0.6'
- **scikit_learn:** '1.9.1'
- **matplotlib:** '3.11.2'
- **streamlit:** '1.66.1'
- **pytest:** '9.1.1'
- **modelo:** 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
- **quantidade_frases_referencia:** 90
- **quantidade_perguntas_medidas:** 30
- **repeticoes:** 30
- **aquecimentos:** 3

## Medições fora da inferência

- **Inicialização do modelo:** 6603.499500 ms
- **Preparação das referências:** 259.379700 ms

## Resumo estatístico

| id_execucao | configuracao | fase | n_execucoes | media_ms | mediana_ms | desvio_padrao_ms | minimo_ms | maximo_ms | p95_ms | threshold | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark_20261026_011428_065614_4b5e8d03 | palavras_chave | primeira_inferencia | 1 | 0.036799996451009065 | 0.036799996451009065 | 0.0 | 0.036799996451009065 | 0.036799996451009065 |  |  | executado |
| benchmark_20261026_011428_065614_4b5e8d03 | embeddings_padrao | primeira_inferencia | 1 | 28.134599997429177 | 28.134599997429177 | 0.0 | 28.134599997429177 | 28.134599997429177 |  |  | executado |
| benchmark_20261026_011428_065614_4b5e8d03 | embeddings_rejeicao_0_50 | primeira_inferencia | 1 | 14.868199999909848 | 14.868199999909848 | 0.0 | 14.868199999909848 | 14.868199999909848 |  | 0.5 | executado |
| benchmark_20261026_011428_065614_4b5e8d03 | palavras_chave | subsequente | 900 | 0.01629555559096237 | 0.01679999695625156 | 0.004006035771390394 | 0.007999995432328433 | 0.05919999966863543 | 0.021315002231858657 |  | executado |
| benchmark_20261026_011428_065614_4b5e8d03 | embeddings_padrao | subsequente | 900 | 19.514245333306867 | 17.31314999778988 | 8.932556415569348 | 12.970099996891804 | 162.40210000250954 | 29.33321499913288 |  | executado |
| benchmark_20261026_011428_065614_4b5e8d03 | embeddings_rejeicao_0_50 | subsequente | 900 | 15.701110333482372 | 15.372400001069764 | 2.454190137013913 | 13.243299996247515 | 54.61690000083763 | 17.6822999947035 | 0.5 | executado |

A primeira inferência permanece separada das execuções subsequentes. O P95 é omitido quando há apenas uma observação.

## Artefatos da execução

- `ambiente`: `ambiente_experimental.json`
- `amostra`: `amostra_perguntas.csv`
- `execucoes`: `latencia_execucoes.csv`
- `grafico`: `grafico_latencia.svg`
- `resumo`: `latencia_resumo.csv`
