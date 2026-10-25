# Avaliação funcional das dicas contextuais

Esta avaliação é independente de `data/frases_referencia.csv` e `data/teste_manual.csv`, que permanecem inalterados. O conjunto de verificação está em `data/avaliacao_funcional_dicas.csv`.

## Critérios

A avaliação é feita em duas camadas:

1. **Classificação:** registrar pergunta, token esperado, token previsto, similaridade e acerto. Essa camada depende do `PuzzleClassifier` real e do modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, usando `MODO_REJEICAO` e threshold `0.50`.
2. **Adequação da dica:** com o token previsto, registrar o estado, a subintenção identificada, se o assunto foi preservado, se a dica respeitou a progressão e se evitou revelar a solução. Os testes determinísticos de `tests/test_contextual_hints.py` exercitam essa camada com classificadores simulados.

O resultado da seleção não é considerado correto apenas porque o token está correto. A verificação descrita aqui é **avaliação funcional automatizada**, não substitui revisão humana de pertinência.

## Casos

O arquivo CSV contém 15 perguntas: seis sobre o cofre, duas sobre o relógio, duas sobre o mural, duas sobre a enciclopédia, duas sobre símbolos e uma fora do escopo. Os casos de senha, conteúdo e relação entre pistas são repetidos em estados distintos pelos testes parametrizados.

| Grupo | Casos | Verificação principal |
|---|---:|---|
| Cofre | 1–6 | senha, conteúdo, finalidade, abertura e não revelação |
| Relógio | 3 e 7 | observação e relação com combinação após progresso |
| Mural | 8–9 | contagem e parte ausente |
| Enciclopédia | 10–11 | associação, tradução e conteúdo |
| Símbolos | 12–14 | significado, ordem e uso na porta |
| Fora do escopo | 15 | rejeição mantida pelo token recebido |

## Estados verificados

- partida inicial;
- cofre aberto;
- fotografia encontrada;
- mural completo;
- porta aberta.

As perguntas explícitas continuam no mesmo assunto em todos esses estados. O estado só altera a variante segura: por exemplo, conteúdo do cofre não confirma a fotografia antes da abertura, enquanto uma pergunta sobre a senha continua apontando para o relógio depois da vitória.

## Registro da execução

A classificação real deve ser executada separadamente da suíte com mocks. Quando o modelo não estiver disponível, o relatório deve marcar a camada de classificação como não executada, sem transformar os resultados dos mocks em evidência do modelo real. Eventuais erros devem ser separados em **erro de classificação** e **erro de seleção contextual**.
