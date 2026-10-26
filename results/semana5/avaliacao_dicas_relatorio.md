# Avaliação funcional das dicas — Semana 5

Fonte: `data/avaliacao_funcional_dicas_resultados.csv`. O CSV original não foi sobrescrito; `avaliacao_dicas_por_cenario.csv` acrescenta apenas uma coluna derivada para análise.

**Registros:** 27 cenários; **perguntas distintas:** 15; **problemas de consistência automática:** 0.

## Métricas

| metrica | numerador | denominador | proporcao | unidade | definicao |
| --- | --- | --- | --- | --- | --- |
| classificacao_por_pergunta_distinta | 10 | 15 | 0.6666666666666666 | pergunta | Uma pergunta é considerada correta somente se todas as suas ocorrências em estados apresentarem token correto. |
| classificacao_por_cenario | 20 | 27 | 0.7407407407407407 | cenário | Cada uma das 27 linhas é uma avaliação de estado; perguntas repetidas não são independentes. |
| dicas_adequadas | 20 | 27 | 0.7407407407407407 | cenário | Julgamento registrado pelo avaliador humano na coluna dica_adequada. |
| sucesso_completo | 18 | 27 | 0.6666666666666666 | cenário | classificacao_correta == Sim e dica_adequada == Sim. |
| respeito_ao_progresso | 27 | 27 | 1.0 | cenário | Julgamento registrado em respeita_progresso. |
| ausencia_de_spoiler | 27 | 27 | 1.0 | cenário | Julgamento registrado em sem_spoiler. |
| token_errado_dica_inadequada | 5 | 27 | 0.18518518518518517 | cenário | Cruzamento entre o token classificado e o julgamento da dica. |
| token_errado_dica_adequada | 2 | 27 | 0.07407407407407407 | cenário | Cruzamento entre o token classificado e o julgamento da dica. |
| token_correto_dica_inadequada | 2 | 27 | 0.07407407407407407 | cenário | Cruzamento entre o token classificado e o julgamento da dica. |
| token_correto_dica_adequada | 18 | 27 | 0.6666666666666666 | cenário | Cruzamento entre o token classificado e o julgamento da dica. |

As métricas por pergunta distinta tratam uma pergunta como correta apenas quando todas as suas repetições de estado estão corretas. A métrica por cenário mantém as 27 linhas e explicita a repetição.

## Quadrantes de classificação e dica

A dica útil não elimina um erro de token, e token correto não garante pertinência da dica. Os quatro quadrantes estão registrados em `avaliacao_dicas_metricas.csv`.

## Casos AV014 e AV020

- AV014: assunto_adequado='Sim', dica_adequada='Sim'; julgamento preservado.
- AV020: assunto_adequado='Sim', dica_adequada='Sim'; julgamento preservado.

Os campos humanos foram preservados. Uma possível interpretação é que uma dica possa ser útil apesar do token errado; outra é exigir que a adequação global dependa também do assunto correto. O CSV atual permanece a autoridade do julgamento e qualquer revisão deve ser documentada pelo avaliador.

## Consistência

Nenhuma inconsistência estrutural ou de coerência automática foi encontrada.
