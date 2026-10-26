# Relatório de integridade dos dados — Semana 5

A verificação é somente leitura. A normalização foi aplicada exclusivamente às comparações e nenhum dataset foi modificado.

**Resultado geral:** APROVADO

## Arquivos e hashes SHA-256

| Arquivo | SHA-256 |
|---|---|
| `data\frases_referencia.csv` | `943cb34419221415de9bc6cbf040a6a1a3e20fbd92269f4df9d47add0ee0a161` |
| `data\teste_manual.csv` | `3ddb6b46d4a3032c6d1b4723913eb0ccbbf1b5ff470fade0ff6b502f527443d5` |
| `data\avaliacao_funcional_dicas.csv` | `dcc24cc0e7cca841f346a8af1f5a502902472997f97bc274dac97f9f7c2dbbb6` |
| `data\avaliacao_funcional_dicas_resultados.csv` | `e8d4bc21788081ce42819207a82ae9ccd9b7adcd589af813529b255b3885bcf9` |
| `data\puzzle_config.json` | `bee2226292f666743d6dab69777d3493ff4f3a00da6ec38d508461312346012b` |
| `results\particao_validacao.csv` | `645607ac1b5c578c89a9c5b425ef5420c3ccd52350389e53dd178f8eb8d8915c` |
| `results\particao_teste.csv` | `b989127dfed6baf517a0e5257007d5e1a8b4573f01acb8c5a193c5a34e67ae69` |
| `results\particao_resumo.csv` | `cc3f93ab25c8c336d62cc3ad4daec3585e41968dfc54c48b959aadf1369b6a2f` |

## Referências semânticas

- Total: **90**; esperado: 90.
- Distribuição: `{'TOKEN_SIMBOLOS': 15, 'TOKEN_LIVRO': 15, 'TOKEN_MURAL': 15, 'TOKEN_COFRE': 15, 'TOKEN_RELOGIO': 15, 'TOKEN_NAO_RELACIONADO': 15}`.
- Duplicatas normalizadas: **0**.

## Perguntas de avaliação

- Total: **100**; esperado: 100.
- Distribuição: `{'TOKEN_SIMBOLOS': 15, 'TOKEN_LIVRO': 15, 'TOKEN_MURAL': 15, 'TOKEN_COFRE': 15, 'TOKEN_RELOGIO': 15, 'TOKEN_NAO_RELACIONADO': 25}`.
- Duplicatas normalizadas: **0**.
- Frases idênticas entre referências e avaliação: **0**.

## Divisão validação/teste

- Semente: **42**; validação: **40%**.
- Tamanhos: validação **40**, teste **60**.
- Sobreposição normalizada: **0**.
- Reprodução determinística: **sim**.
- Compatibilidade dos arquivos gerados: `{'validacao': {'arquivo': 'C:\\Users\\Kleber\\Desktop\\TCC\\Repositório\\results\\particao_validacao.csv', 'presente': True, 'linhas_esperadas': 40, 'linhas_geradas': 40, 'compatível': True}, 'teste': {'arquivo': 'C:\\Users\\Kleber\\Desktop\\TCC\\Repositório\\results\\particao_teste.csv', 'presente': True, 'linhas_esperadas': 60, 'linhas_geradas': 60, 'compatível': True}}`.

## Candidatos a revisão de ambiguidade

Os itens abaixo foram encontrados por termos de objetos relacionados. Eles são apenas alertas; não constituem decisão de anotação.

| Pergunta | Rótulo | Tokens sugeridos por termos |
|---|---|---|
| Será que aquilo escrito perto da saída mostra a ordem da combinação? | `TOKEN_SIMBOLOS` | TOKEN_SIMBOLOS, TOKEN_COFRE |
| Os sinais desenhados junto à entrada revelam quais teclas usar? | `TOKEN_SIMBOLOS` | TOKEN_SIMBOLOS, TOKEN_COFRE |
| Devo procurar uma tradução dentro das páginas sobre bichos? | `TOKEN_LIVRO` | TOKEN_MURAL, TOKEN_LIVRO |
| O livro de animais serve para descobrir o nome de cada símbolo? | `TOKEN_LIVRO` | TOKEN_SIMBOLOS, TOKEN_MURAL, TOKEN_LIVRO |
| Preciso folhear a enciclopédia até achar o bicho certo? | `TOKEN_LIVRO` | TOKEN_MURAL, TOKEN_LIVRO |
| Há algo útil nas páginas do livro sobre a escrita da parede? | `TOKEN_LIVRO` | TOKEN_SIMBOLOS, TOKEN_LIVRO |
| Talvez essa coleção de animais funcione como um dicionário? | `TOKEN_LIVRO` | TOKEN_MURAL, TOKEN_LIVRO |
| A obra ilustrada pode ensinar a correspondência entre criaturas e símbolos? | `TOKEN_LIVRO` | TOKEN_SIMBOLOS, TOKEN_MURAL, TOKEN_LIVRO |
| Como eu descubro a correspondência entre os nomes dos animais e aquela escrita? | `TOKEN_LIVRO` | TOKEN_MURAL, TOKEN_LIVRO |
| A caixa trancada contém a parte que falta da investigação? | `TOKEN_COFRE` | TOKEN_COFRE, TOKEN_MURAL |
| Vale a pena destrancar o cofre antes de mexer na porta? | `TOKEN_COFRE` | TOKEN_SIMBOLOS, TOKEN_COFRE |
| Esse horário congelado pode ser uma combinação? | `TOKEN_RELOGIO` | TOKEN_COFRE, TOKEN_RELOGIO |
| O relógio que não funciona mais esconde algum código? | `TOKEN_RELOGIO` | TOKEN_COFRE, TOKEN_RELOGIO |
| As horas indicadas no relógio servem para o cofre? | `TOKEN_RELOGIO` | TOKEN_COFRE, TOKEN_RELOGIO |
| A marcação temporal fixa tem utilidade para a caixa trancada? | `TOKEN_RELOGIO` | TOKEN_SIMBOLOS, TOKEN_COFRE, TOKEN_RELOGIO |
| Como transformar o instante exibido em uma senha? | `TOKEN_RELOGIO` | TOKEN_COFRE, TOKEN_RELOGIO |

## Pendências

Nenhuma inconsistência automática encontrada.
