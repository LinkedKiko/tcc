# Protocolo de reprodução — Semana 5

Todos os comandos partem da raiz do projeto. A Semana 5 organiza a evidência sem alterar `data/frases_referencia.csv`, `data/teste_manual.csv`, `data/puzzle_config.json`, `data/avaliacao_funcional_dicas_resultados.csv`, os classificadores ou as regras do puzzle.

## 1. Ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Para repetir o ambiente registrado, use as versões de `requirements-lock.txt`. A identificação observada durante a execução fica nos manifestos em `results/semana5/`; modelos baixados e `.venv/` não fazem parte da entrega. O modelo é `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. A revisão do modelo não estava disponível no registro local; por isso o nome completo e as versões das bibliotecas são registrados, sem inventar um hash de revisão.

## 2. Integridade, dados e históricos

```powershell
python scripts/validar_integridade_semana5.py
python scripts/consolidar_resultados_semana5.py
```

A auditoria confirma as colunas, 90 referências, 100 perguntas, distribuição dos seis tokens, ausência de duplicatas normalizadas e divisão determinística de 40 perguntas de validação e 60 de teste com `RANDOM_STATE = 42`. Os hashes estão em `results/semana5/integridade_dados.json`.

`results/semana5/reexecucao_semana2/` é histórico cuja procedência original foi registrada por hashes; arquivos presentes não são, sozinhos, prova de execução completa. Os resultados científicos históricos permanecem em `results/` e não devem ser sobrescritos.

## 3. Reprodução verificável da Semana 2

Use o registrador para capturar comando, diretório, Python, bibliotecas, horários, duração, código de saída, commit, hashes e arquivos gerados:

```powershell
python scripts/executar_reproducao_semana5.py
```

Internamente, o comando experimental é:

```powershell
python evaluate.py --modo completo --resultados results/semana5/reproducao_final --sem-graficos
```

O modo `completo` seleciona o threshold somente na validação, aplica o valor ao teste uma única vez e grava as três configurações. O resultado deve ser classificado como `reprodução concluída`, `reprodução concluída com divergências` ou `reprodução incompleta` conforme código de saída e arquivos obrigatórios, nunca apenas pela existência de CSVs. O relatório final é `results/semana5/relatorio_reprodutibilidade.md` e o manifesto fica em `results/semana5/reproducao_final/manifesto_execucao.json`.

## 4. Benchmark estratificado

```powershell
python scripts/benchmark_latencia.py --por-classe 5 --seed 42 --repeticoes 30 --aquecimentos 3
```

A função `amostrar_perguntas_estratificadas` usa `token_esperado` e `frase`, ordena candidatos de forma canônica para não depender da ordem do CSV, amostra sem reposição com seed 42 e registra `id_pergunta`, texto, token, posição original e seed. O resultado tem 30 perguntas: cinco para cada classe existente (`TOKEN_SIMBOLOS`, `TOKEN_LIVRO`, `TOKEN_MURAL`, `TOKEN_COFRE`, `TOKEN_RELOGIO` e `TOKEN_NAO_RELACIONADO`). Uma classe com menos exemplos do que `--por-classe` causa erro explicativo.

Cada execução cria:

```text
results/semana5/benchmark_final/
└── execucao_<id_execucao>/
    ├── amostra_perguntas.csv
    ├── latencia_execucoes.csv
    ├── latencia_resumo.csv
    ├── ambiente_experimental.json
    ├── relatorio_latencia.md
    ├── grafico_latencia.svg
    └── manifesto_execucao.json
```

O mesmo `id_execucao` aparece no CSV individual, resumo, JSON, Markdown e manifesto. Os três classificadores usam exatamente a amostra registrada. A inicialização do MiniLM e a preparação das referências são medidas separadamente; a primeira inferência fica separada das 900 inferências subsequentes por configuração (`30 perguntas × 30 repetições`). Os aquecimentos são três por configuração e não entram nas estatísticas. A escrita dos arquivos ocorre fora de cada intervalo de `time.perf_counter()`.

O argumento legado `--max-perguntas` não trunca mais o CSV. Ele só é aceito para total múltiplo das seis classes e é convertido em quantidade igual por classe; portanto `--max-perguntas 15` é rejeitado para impedir a reprodução silenciosa da amostra enviesada histórica.

A medição anterior, com 15 perguntas do prefixo `data/teste_manual.csv`, é preservada como **benchmark preliminar de uma única classe** nos arquivos antigos da raiz de `results/semana5/`. Ela não deve ser misturada ao diretório `benchmark_final`.

## 5. Testes automatizados

```powershell
python scripts/registrar_testes_semana5.py
```

O registrador executa, de forma separada:

```powershell
python -m pytest -q --junitxml=results/semana5/testes_finais/pytest_completo.xml
python -m pytest tests/test_streamlit_app.py -v
python -m pytest tests/test_integracao_modelo.py -m integracao -v
python -m pytest tests/test_benchmark_latencia.py tests/test_consolidacao_semana5.py tests/test_integridade_semana5.py -v
```

Os logs são `pytest_completo.out`, `pytest_streamlit.out`, `pytest_integracao_modelo.out` e `pytest_semana5.out`. `resumo_testes.json` e `relatorio_testes.md` registram coleta, aprovados, falhas, ignorados, erros, duração e código de saída. A quantidade global vem somente do XML da suíte completa; execuções direcionadas não são somadas.

O teste de integração carrega efetivamente o MiniLM. Se houver falha de download, dependência ausente ou interrupção, o registro deve dizer falhou, foi ignorado ou ficou incompleto; não deve convertê-la em aprovação.

## 6. Interface e jogo

```powershell
python -m streamlit run streamlit_app.py
```

A validação automatizada de `tests/test_streamlit_app.py` cobre a instância da engine, isolamento de sessões, fluxo dos objetos, threshold, dicas, reinício e vitória. A execução interativa não é declarada como evidência automática apenas por importar o módulo.

## 7. Documentação e rastreabilidade

- `README.md`: objetivo, tecnologias, arquitetura, execução, avaliação, latência, testes, limitações e versão experimental.
- `docs/resultados_discussao_semana5.md`: métricas históricas, benchmark final, testes e divergências, usando os valores da execução identificada.
- `results/semana5/versao_final.json`: commit, tag, hashes dos datasets, execução do benchmark, reprodução e testes.
- `requirements.txt`: dependências de instalação.
- `requirements-lock.txt`: versões efetivamente registradas, sem caminhos locais ou credenciais.

Antes do encerramento Git:

```powershell
git status --short
git log -1
git tag --list
```

Somente após validar os artefatos, fazer commit local específico e criar a tag `tcc-semana5-final` se ela ainda não existir. Nenhum push remoto é feito por este protocolo.
