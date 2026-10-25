# Correções finais da Semana 4

## 1. Escopo

Foi feita uma revisão incremental da implementação existente. `PuzzleEngine`, `PuzzleState`, `PuzzleConfig`, o classificador e os datasets experimentais não foram reimplementados nem alterados. A configuração continua derivando as combinações e totais de `data/puzzle_config.json`.

## 2. Problemas identificados e causas

- A seleção contextual de `TOKEN_COFRE` consultava o progresso antes de respeitar perguntas explícitas sobre senha, abertura e conteúdo. Depois de abrir o cofre, perguntas sobre a senha podiam receber uma orientação sobre fotografia ou mural.
- `TOKEN_RELOGIO`, `TOKEN_MURAL`, `TOKEN_LIVRO` e `TOKEN_SIMBOLOS` também possuíam fallbacks de progresso que substituíam assuntos explícitos, como combinação, contagem, associação e ordem.
- O teste antigo exigia a palavra `fotografia` para conteúdo com o cofre fechado, em conflito com a política de não revelar antecipadamente o conteúdo.
- A interface renderizava indicadores antes de ações mutáveis sem rerun controlado; a ação era salva, mas a atualização visual podia ficar para a próxima interação.
- `st.balloons()` era chamado em toda execução enquanto a porta permanecesse aberta.
- Falhas do classificador exibiam o texto bruto da exceção na interface.

## 3. Alterações realizadas

### Seleção de dicas

`src/game/hint_manager.py` passou a identificar a subintenção antes de consultar o estado:

```text
TOKEN + pergunta → subintenção → estado → variante cadastrada
```

As regras usam normalização de acentos e expressões/palavras completas. O token principal recebido do classificador nunca é alterado por essas regras.

- Cofre: senha, abertura, conteúdo e finalidade são diferenciados. Senha continua apontando para o relógio em todos os estados; abertura continua explicando o procedimento; conteúdo só confirma a fotografia depois da abertura; finalidade adapta a progressão sem revelar antes da hora.
- Relógio: perguntas sobre combinação continuam relacionadas ao cofre após a combinação ter sido usada.
- Mural: perguntas sobre animais, contagem, parte ausente e valores preservam esses assuntos após o mural estar completo.
- Livro: perguntas sobre conteúdo, tradução, associação e utilização continuam associadas à enciclopédia mesmo após a conclusão do mural.
- Símbolos: perguntas sobre significado, associação, ordem e uso na porta não são substituídas pelo fallback de progresso.
- `TOKEN_NAO_RELACIONADO` continua usando a resposta de rejeição cadastrada; nenhuma palavra da pergunta redireciona esse token.
- Instâncias de `HintManager` com dicas customizadas preservam o comportamento legado.

### Política de não revelação

Antes da abertura, a pergunta sobre conteúdo recebe orientação para descobrir a combinação e abrir o cofre, sem confirmar a fotografia. Depois da abertura, a dica pode orientar a busca/exame da fotografia. As combinações `091530` e `2534` não aparecem nas dicas.

### Streamlit

`streamlit_app.py` agora:

- registra o resultado antes de `st.rerun()` nas ações que alteram cofre, fotografia, mural e porta;
- mantém a mensagem em `st.session_state`, de modo que o rerun reflita o novo estado sem repetir a ação;
- conserva uma engine por sessão e o classificador em `st.cache_resource`;
- usa `VICTORY_ANIMATED_KEY` para executar a animação uma vez por partida;
- mantém a mensagem de vitória em reruns posteriores;
- restaura a flag no reinício;
- registra detalhes técnicos no log e exibe uma mensagem segura ao jogador quando o classificador falha.

## 4. Distinção metodológica

**Classificação da intenção** é a etapa semântica: o `PuzzleClassifier` carrega o modelo MiniLM, calcula similaridade de cosseno e produz um token, usando modo e threshold recebidos pela interface.

**Seleção contextual da dica** é uma etapa determinística posterior: o `HintManager` recebe o token já produzido, a pergunta e um snapshot do estado, identifica uma subintenção por regras controladas e escolhe uma mensagem predefinida. Não é um segundo modelo de embeddings, não treina parâmetros e não gera texto livre.

## 5. Testes e verificações executadas

| Verificação | Resultado |
|---|---:|
| `tests/test_contextual_hints.py` com `--noconftest` e plugins internos | 71 aprovados |
| `tests/test_game_integration.py` com plugins internos | 4 aprovados |
| `tests/test_puzzle_engine.py` | 12 aprovados |
| `tests/test_streamlit_app.py` sem o smoke AppTest | 14 aprovados |
| Validação manual dos casos contextuais | aprovada |
| Validação de sintaxe de código e testes alterados | aprovada |
| Smoke `AppTest` | aprovado no teste de inicialização da interface |
| Integração com MiniLM real | não concluída: importação de `sklearn.metrics.pairwise` bloqueou no ambiente |

Os números acima não incluem testes marcados como ignorados. O ambiente global e o `.venv` usam Python 3.14.7 e `scikit-learn` 1.9.1; a instalação das dependências foi tentada no `.venv`, mas a importação do módulo de similaridade não completou. Por isso não há afirmação de que o modelo real foi carregado.

A validação manual verificou as prioridades em cinco estados, uma chamada do classificador por pergunta, ausência de mutação de estado, rejeição preservada, cache por instância e fluxo completo do motor. Ela não substitui a integração real nem o smoke executado pelo runner do Streamlit.

## 6. Avaliação funcional

A avaliação independente das métricas da Semana 2 está em `data/avaliacao_funcional_dicas.csv` e `docs/avaliacao_funcional_dicas.md`. São 15 perguntas abrangendo cofre, relógio, mural, enciclopédia, símbolos e uma pergunta fora do escopo. Os testes contextuais exercitam os mesmos assuntos nos estados inicial, cofre aberto, fotografia encontrada, mural completo e porta aberta.

O critério separa token incorreto — erro de classificação — de token correto com dica inadequada — erro de seleção contextual. O resultado local da camada determinística foi aprovado pelos 64 testes contextuais; a camada real de classificação não pôde ser registrada por causa da limitação de `scikit-learn` descrita acima.

## 7. Evidências visuais

As capturas reais da aplicação estão disponíveis em `screenshots/`. O inventário documentado em `screenshots/README.md` inclui tela inicial, relógio, cofre fechado e aberto, fotografia, mural completo, enciclopédia, símbolos, uma captura de dica contextual do cofre, vitória e reinício.

As imagens foram mantidas separadas dos datasets e resultados experimentais. Elas servem como evidência visual da interface executada e complementam, sem substituir, os testes automatizados e a validação do modelo real.

## 8. Preservação do experimento da Semana 2

Não foram alterados `data/frases_referencia.csv`, `data/teste_manual.csv` ou os arquivos existentes em `results/`. Também não foram modificados o modelo configurado, o cálculo de similaridade, o particionamento ou o threshold experimental. A nova avaliação funcional é separada e não deve ser misturada às métricas originais.

## 9. Pendências objetivas

1. Reexecutar `pytest -q` em um ambiente Python compatível no qual os módulos binários de `scikit-learn` importem normalmente.
2. Executar `pytest -m integracao` e registrar tokens, similaridades e dicas com o modelo MiniLM real.
3. Executar o smoke/interações `AppTest` para complementar as evidências visuais já disponíveis.
4. Registrar, se necessário, uma validação independente dos reruns, indicadores laterais e animação única; as capturas correspondentes já estão em `screenshots/`.

## 10. Correção da divulgação progressiva do mural

A revisão complementar corrigiu a revelação antecipada da fotografia. A antiga mensagem da interface — `Encontre a fotografia no cofre para completar a parte ausente.` — informava simultaneamente a existência, a localização e a função do objeto antes de ele ser descoberto. Ela foi substituída por orientações de investigação da sala; quando o cofre já está aberto, a mensagem passa a orientar o exame das pistas disponíveis, e somente depois de encontrar a fotografia explica a combinação das duas partes do mural.

As variantes de `TOKEN_MURAL` agora são selecionadas pela combinação determinística de token, subintenção da pergunta e estado da partida. Há mensagens distintas para mural incompleto, cofre aberto, fotografia encontrada e mural completo, além de variantes seguras para contagem e valores numéricos. Perguntas explícitas continuam prioritárias: perguntar sobre contagem permanece uma pergunta sobre contagem, mas recebe o nível de detalhe permitido pelo progresso.

Antes de abrir o cofre, perguntas sobre fotografia ou parte ausente não recebem a localização do objeto. Após a abertura, podem orientar o exame do interior; depois da descoberta, podem explicar sua relação com o mural; com o mural completo, orientam a consulta à enciclopédia e a relação com os símbolos. O motor também deixou de dizer que a fotografia estava dentro do cofre quando um acesso antecipado era tentado.

A interface agora omite o indicador lateral e a seção **Fotografia** no início da partida. Após abrir o cofre, o fluxo existente de **Encontrar fotografia** continua disponível. Os demais objetos e regras de acesso foram preservados.

## 11. Cenários de dicas após a correção

- **Cofre fechado:** “Observe os animais que aparecem na parte visível do mural. Como uma região está danificada, investigue os objetos da sala para reconstruir as informações ausentes.”
- **Cofre aberto:** “Agora que o cofre está aberto, examine seu interior para encontrar a informação que pode completar a parte danificada do mural.”
- **Fotografia encontrada:** “Compare os animais mostrados na fotografia com os que aparecem na parte visível do mural para determinar os totais.”
- **Mural completo:** “Com as quantidades totais dos animais, consulte a enciclopédia e relacione cada total à ordem dos símbolos.”

Essas mensagens são textos cadastrados, sem geração automática, nova classificação ou chamada adicional ao modelo de embeddings.

## 12. Validação complementar

Foram acrescentados testes para as quatro fases de divulgação, perguntas explícitas sobre contagem, números e fotografia, ausência de fotografia na interface inicial, disponibilidade da seção após a abertura, mensagens de acesso antecipado e preservação das quatro flags ao solicitar dicas. Os comandos executados e seus resultados estão registrados no relatório final da tarefa; qualquer limitação ambiental é indicada sem ser tratada como aprovação.

## 13. Resultado da validação desta correção

Foram executados os comandos solicitados:

- `python -m pytest tests/test_contextual_hints.py -v` — **71 passed**;
- `python -m pytest tests/test_streamlit_app.py -v` — **14 passed**, incluindo o smoke test do Streamlit;
- `python -m pytest -q` — **202 passed em 14,46 s**.

A execução do servidor interativo não foi mantida pelo agente, pois `streamlit run` é um processo de longa duração; o comando para validação manual continua sendo `python -m streamlit run streamlit_app.py`. A funcionalidade equivalente de inicialização foi verificada pelo `AppTest`, e o fluxo completo do motor continua coberto pela suíte. A integração com o modelo real permanece sujeita à disponibilidade das dependências/modelo descrita anteriormente.
