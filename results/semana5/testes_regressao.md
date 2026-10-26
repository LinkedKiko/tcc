# Registro histórico de testes e validação — Semana 5

> Este arquivo preserva os registros anteriores da auditoria. Não some suas execuções com outras tentativas. A evidência final, com uma suíte completa única e grupos direcionados separados, está em `results/semana5/testes_finais/resumo_testes.json` e `relatorio_testes.md`: 213/213 na suíte completa, 14/14 no Streamlit, 3/3 na integração real do MiniLM e 11/11 nos testes específicos da Semana 5. A contagem histórica de 205 testes direcionados abaixo não representa a execução final.

## Testes direcionados concluídos

| Arquivo | Aprovados | Falhas | Ignorados | Erros de coleta | Tempo |
|---|---:|---:|---:|---:|---:|
| `tests/test_classifier.py` | 31 | 0 | 0 | 0 | 1,36 s |
| `tests/test_evaluation.py` | 44 | 0 | 0 | 0 | 1,74 s |
| `tests/test_keyword_classifier.py` | 23 | 0 | 0 | 0 | 0,06 s |
| `tests/test_contextual_hints.py` | 71 | 0 | 0 | 0 | registrado no log direcionado |
| `tests/test_puzzle_engine.py` | 12 | 0 | 0 | 0 | 0,07 s |
| `tests/test_game_integration.py` | 4 | 0 | 0 | 0 | 0,04 s |
| `tests/test_streamlit_app.py` | 14 | 0 | 0 | 0 | 1,04 s |
| `tests/test_integridade_semana5.py` + `tests/test_consolidacao_semana5.py` | 6 | 0 | 0 | 0 | 1,29 s |
| **Total direcionado** | **205** | **0** | **0** | **0** | — |

Os logs textuais individuais permanecem em `results/semana5/` quando necessário para auditoria. Os testes de Streamlit incluem o smoke da sala, cache do classificador, reinício, vitória e preservação das flags de estado.

## Integração com modelo real

Foi executado `python -m pytest tests/test_integracao_modelo.py -m integracao -v`. O processo permaneceu na etapa de coleta neste ambiente e não produziu contagem final; portanto esse teste específico é **inconclusivo**, não aprovado nem ignorado. A utilização real do modelo foi, contudo, confirmada independentemente pelo benchmark: o MiniLM foi carregado, 90 referências foram codificadas e 1.350 classificações subsequentes foram medidas.

## Suíte completa

Foi tentado `python -m pytest -q`. O runner do ambiente encerrou a tentativa sem produzir stdout/stderr persistido, portanto não há evidência suficiente para informar uma contagem da suíte completa. Os resultados acima correspondem somente aos arquivos executados e registrados individualmente.

## Smoke funcional adicional

O fluxo programático do `PuzzleEngine` passou por estado inicial, senha incorreta, senha correta derivada da configuração, fotografia, mural, porta incorreta, vitória e reinício. A importação de `streamlit_app` passou. Uma chamada direta de `AppTest` em modo bare emitiu o aviso de ausência de `ScriptRunContext` e não forneceu confirmação adicional; a validação efetiva da interface está representada pelos 14 testes de `tests/test_streamlit_app.py`.

A reexecução integral de `evaluate.py` foi tentada em `results/semana5/reexecucao_semana2`, sem sobrescrever os históricos. As partições foram geradas e o carregamento do modelo iniciou, mas a execução terminou antes das métricas; essa tentativa é registrada como incompleta. A ausência de novos CSVs de métricas impede declarar reprodução integral atualizada.
