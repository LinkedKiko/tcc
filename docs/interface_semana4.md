# Interface da Semana 4

## 1. Objetivo da interface

A Semana 4 acrescenta uma camada visual ao sistema desenvolvido nas semanas anteriores. A aplicação local **Sala dos Símbolos** permite explorar os objetos, inserir as combinações derivadas das pistas, acompanhar o progresso e formular perguntas em português sem transferir regras do puzzle para a interface.

A interface não gera dicas novas. Ela apresenta as dicas determinísticas selecionadas pelo token produzido pelo classificador semântico.

## 2. Tecnologias utilizadas

- Python 3;
- Streamlit para a aplicação web local;
- `PuzzleEngine`, `PuzzleState`, `PuzzleConfig` e `HintManager` da Semana 3;
- `PuzzleClassifier`, Sentence Transformers e similaridade de cosseno das semanas anteriores;
- pandas, NumPy, scikit-learn, matplotlib e pytest, preservados no projeto.

O modelo utilizado continua sendo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Não há treinamento, fine-tuning, banco de dados, autenticação, serviço externo de IA ou geração livre de texto.

## 3. Arquitetura de integração

O fluxo é separado em camadas:

```text
Streamlit
  ├── formulários e botões
  ├── st.session_state (uma engine por sessão)
  └── apresentação dos retornos
          ↓
PuzzleEngine
  ├── PuzzleState: cofre, fotografia, mural e porta
  ├── PuzzleConfig: fatos do JSON e senhas derivadas
  └── HintManager: token → dica cadastrada
          ↓
PuzzleClassifier → embeddings → similaridade de cosseno
```

Cada interação chama um método público existente do `PuzzleEngine`. A interface não calcula senhas, contagens, traduções de símbolos ou classificações paralelas. Os dados exibidos em tabelas são extraídos do campo `dados` devolvido pelo motor.

## 4. Gerenciamento de sessão

O Streamlit reexecuta o script após interações. Para evitar a recriação da partida, `streamlit_app.py` guarda a instância de `PuzzleEngine` em `st.session_state`. A engine contém um `PuzzleState` próprio; portanto, duas sessões de navegador recebem partidas independentes. O botão **Reiniciar jogo** chama `engine.reiniciar()` e limpa resultados transitórios, perguntas, dicas e erros da partida anterior.

O `PuzzleClassifier` é carregado somente quando a primeira pergunta é enviada. A função `carregar_classificador` usa `st.cache_resource`, compartilhando entre sessões apenas o recurso de classificação, que é tratado como somente leitura depois da construção. O estado mutável da partida nunca é colocado no cache. Assim, as 90 referências e o modelo não são recarregados a cada clique.

A sessão não é persistida em disco ou banco de dados: atualizar a página ou encerrar a sessão pode remover o progresso, conforme o comportamento padrão do Streamlit.

## 5. Objetos interativos

- **Relógio:** `engine.examinar_objeto("relogio")`; apresenta o horário configurado como pista, sem mostrar a combinação do cofre.
- **Cofre:** `engine.abrir_cofre(senha)` recebe texto, preservando zeros à esquerda. Depois da abertura, `engine.encontrar_fotografia()` disponibiliza a fotografia.
- **Fotografia:** só é examinável após ser encontrada; suas quantidades são as devolvidas por `engine.examinar_objeto("fotografia")`.
- **Mural:** mostra inicialmente as quantidades visíveis. Após a fotografia, `engine.completar_mural()` calcula os totais pelo motor.
- **Enciclopédia:** `engine.consultar_enciclopedia()` apresenta símbolos e animais, sem inserir valores numéricos.
- **Símbolos:** `engine.examinar_objeto("simbolos")` mostra a sequência da configuração, sem calcular ou exibir a senha final.
- **Porta:** `engine.abrir_porta(senha)` valida a combinação como texto e registra a vitória quando aceita.

As consultas e ações repetidas respeitam as mensagens idempotentes do motor. Senhas incorretas não alteram o estado.

## 6. Classificação semântica

O campo **Precisa de uma dica?** envia a pergunta uma única vez quando o formulário é submetido. O caminho utilizado é:

```python
resultado = engine.processar_pergunta(
    pergunta,
    classificador,
    modo=MODO_REJEICAO,
    threshold=0.50,
)
```

O valor `SIMILARITY_THRESHOLD` em `src/config.py` permanece `None`; o limiar escolhido experimentalmente é aplicado explicitamente somente na camada da interface. Perguntas abaixo de `0.50` recebem `TOKEN_NAO_RELACIONADO` e a mensagem de rejeição já cadastrada. O resultado principal é `resultado["dica"]`; os dados técnicos ficam em **Detalhes da classificação**.

A similaridade de cosseno é apresentada como escore do modelo, nunca como porcentagem ou probabilidade de acerto. A interface também informa falhas de dataset, biblioteca ou carregamento do modelo sem inventar token ou dica.

## 7. Testes

Foi criado `tests/test_streamlit_app.py`, que usa classificadores falsos para evitar o download do modelo real. Os testes verificam inicialização e persistência da engine, isolamento entre sessões, fluxo de senhas e objetos, fotografia, mural, enciclopédia, símbolos, porta, dicas, rejeição, preservação do estado ao perguntar, reinício e reutilização do classificador cacheado. Também há um smoke test com `streamlit.testing.v1.AppTest` para confirmar a inicialização da tela e a presença dos objetos.

Além desses testes, a suíte original (`python -m pytest -q`) deve continuar sendo executada. A validação real do modelo é separada em `tests/test_integracao_modelo.py` e depende da instalação e disponibilidade do modelo.

A verificação manual recomendada é iniciar a aplicação com `python -m streamlit run streamlit_app.py`, abrir o relógio, testar uma senha inválida e a combinação derivada, encontrar a fotografia, completar o mural, consultar enciclopédia e símbolos, testar a porta, enviar perguntas relacionadas e fora do escopo e reiniciar.

## 8. Limitações

- O primeiro carregamento do modelo pode ser demorado e requer o download do modelo quando ele não estiver em cache.
- O desempenho das respostas depende das frases de referência, do modelo e do threshold selecionado na validação; o escore não é uma probabilidade.
- O conjunto de dicas é estático e não possui personalização por jogador ou histórico.
- A interface usa componentes nativos do Streamlit e é deliberadamente simples; não há imagens essenciais, animações complexas ou inventário.
- O progresso é mantido somente na sessão do navegador e não é persistido após o encerramento.
- A engine atual permite validar a porta sem exigir que todas as ações anteriores tenham sido executadas; a interface orienta a exploração, mas não cria uma regra adicional diferente do motor.
- As capturas reais da execução estão disponíveis em `screenshots/`, com inventário em `screenshots/README.md`.


## 9. Correções da revisão final

A seleção contextual agora segue a prioridade pergunta explícita → subintenção → estado → variante cadastrada. A pergunta sobre senha ou combinação do cofre continua recebendo a orientação do relógio em qualquer estado; perguntas sobre conteúdo e finalidade seguem variantes próprias e respeitam a não revelação antecipada. As regras equivalentes foram revisadas para relógio, mural, enciclopédia e símbolos. O token de rejeição é retornado diretamente ao mapa de respostas, sem substituição lexical.

Ações mutáveis da interface registram o retorno da engine antes de solicitar `st.rerun()`. Como a engine permanece no `st.session_state`, o rerun atualiza os indicadores laterais sem repetir a ação nem recriar o motor. A vitória continua visível, mas `st.balloons()` é protegido por uma flag de sessão e só volta a ser permitido após reiniciar.

A função `carregar_classificador` continua sob `st.cache_resource`; o modelo e os embeddings de referência são reutilizados, e somente a engine com seu `PuzzleState` é isolada por sessão. Falhas de classificação são registradas no log e apresentadas por mensagem compreensível, sem expor traceback ou caminhos internos.

A avaliação funcional automatizada está separada dos resultados científicos da Semana 2 em `docs/avaliacao_funcional_dicas.md` e `data/avaliacao_funcional_dicas.csv`. Ela distingue erro de classificação — token real incorreto — de erro contextual — token correto, mas variante inadequada. A suíte com mocks não é evidência de carregamento do modelo real.

## 10. Evidências e limitações

A suíte unitária e o smoke test do Streamlit devem ser executados no ambiente do projeto. A integração real requer que Sentence Transformers esteja instalado e que o modelo obrigatório esteja disponível localmente ou possa ser baixado. Se essa condição não ocorrer, o resultado deve ser registrado como limitação, não como aprovação.

As capturas de tela disponíveis em `screenshots/` foram obtidas exclusivamente executando a aplicação real; o diretório não contém imagens fictícias. O conjunto cobre tela inicial, estados intermediários, uma situação de dica do cofre, vitória e estado após reinício.


## 11. Capturas de tela produzidas

Foram registradas capturas reais da interface em `screenshots/`: tela inicial (`01-inicial.png`), relógio (`02-relogio.png`), cofre fechado e aberto (`03-cofre-fechado.png` e `04-cofre-aberto.png`), fotografia (`05-fotografia.png`), mural completo (`06-mural-completo.png`), enciclopédia (`07-enciclopedia.png`), símbolos (`08-simbolos.png`), uma captura de dica do cofre (`09-dica.png`), vitória (`10-vitoria.png`) e reinício (`11-reset.png`).

Essas imagens constituem evidência visual da execução da interface. A validação automatizada do `AppTest` e a integração com o modelo real continuam sendo resultados independentes e devem ser interpretadas conforme as limitações registradas em `docs/correcoes_semana4.md`.

## 12. Política de divulgação progressiva das pistas

A interface segue uma política de descoberta progressiva. No estado inicial, o mural mostra somente que existe uma parte danificada e orienta a investigação dos objetos; o nome e o indicador da fotografia não aparecem na barra lateral nem como seção visual. Isso preserva a observação e a dedução necessárias para descobrir a próxima pista.

Quando `cofre_aberto` passa a ser verdadeiro, a seção **Fotografia** torna-se visível e o botão existente **Encontrar fotografia** continua sendo o caminho para disponibilizá-la. Quando `foto_encontrada` é verdadeiro, a interface permite examiná-la e o mural pode ser completado. Depois de `mural_completo`, os totais podem ser relacionados à enciclopédia e à sequência dos símbolos. Nenhuma dessas mudanças altera os métodos públicos ou as regras de acesso do motor.

O `HintManager` aplica a mesma política depois da classificação: `token + pergunta + snapshot de PuzzleState → subintenção → variante cadastrada`. A pergunta explícita tem prioridade sobre o fallback de progresso. Assim, “Como devo contar os animais?” continua produzindo uma orientação de contagem, enquanto o estado escolhe entre contar a parte visível, combinar com a fotografia ou relacionar os totais à enciclopédia. Perguntas que mencionam fotografia antes da descoberta recebem uma orientação segura sem confirmar localização ou função; após abrir o cofre, podem mencionar o exame da pista e, após encontrá-la, sua relação com o mural.

A apresentação também é defensiva: `_renderizar_fotografia()` não renderiza a seção enquanto o cofre estiver fechado, `_mostrar_estado()` omite o indicador nesse estado e `_renderizar_mural()` usa textos que não identificam antecipadamente o conteúdo oculto. O motor mantém as transições, eventos, cálculos, senhas e estado originais.

## 13. Testes da política

Os testes contextuais cobrem mural inicial, contagem inicial, cofre aberto, fotografia encontrada, mural completo, senha do cofre em todos os estados e imutabilidade de `cofre_aberto`, `foto_encontrada`, `mural_completo` e `porta_aberta`. Os testes da interface cobrem a ausência da fotografia na tela inicial, sua renderização somente após a abertura, o indicador lateral e o fluxo normal dos objetos. A validação do modelo MiniLM continua separada e não é substituída por esses testes determinísticos.

## 14. Resultado da validação desta correção

A validação executada foi:

- `python -m pytest tests/test_contextual_hints.py -v`: 71 testes aprovados;
- `python -m pytest tests/test_streamlit_app.py -v`: 14 testes aprovados, incluindo a inicialização por `AppTest`;
- `python -m pytest -q`: 202 testes aprovados em 14,46 segundos.

Para inspeção manual, iniciar com `python -m streamlit run streamlit_app.py`. O agente não mantém esse processo interativo em execução, mas a inicialização foi verificada automaticamente pelo smoke test e as transições de cofre, fotografia, mural e porta foram verificadas pela suíte.
