# Relatório histórico de latência — Semana 5

> **Artefato preliminar, não definitivo.** Esta pasta raiz preserva a medição antiga com 15 perguntas do prefixo do CSV. O `relatorio_latencia.md` e o `ambiente_experimental.json` foram produzidos em execuções diferentes e apresentam valores divergentes; por isso este relatório não é usado como resultado final. A medição definitiva está em `benchmark_final/execucao_benchmark_20261026_011428_065614_4b5e8d03/`, cujo JSON, CSVs, gráfico, manifesto e Markdown compartilham o mesmo `id_execucao`.

A medição foi executada pelo script `scripts/benchmark_latencia.py`, reutilizando `PuzzleClassifier` e `KeywordClassifier`. Foram utilizadas as 15 primeiras perguntas, na ordem original de `data/teste_manual.csv`, 3 aquecimentos por configuração e 30 repetições por configuração. A primeira inferência foi registrada separadamente; as 450 observações subsequentes de cada configuração correspondem a 30 × 15.

## Ambiente

- Sistema operacional: Windows 11 (`Windows-11-10.0.26200-SP0`)
- Python: 3.14.7
- Processador: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD; 12 CPUs lógicas
- Dispositivo: CPU; CUDA indisponível
- Sentence Transformers: 6.1.0
- PyTorch: 2.14.0
- NumPy: 2.5.3; pandas: 3.0.6; scikit-learn: 1.9.1; matplotlib: 3.11.2; Streamlit: 1.66.1; pytest: 9.1.1
- Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Referências: 90; perguntas medidas: 15
- Inicialização do modelo: **6.191,61 ms**
- Preparação isolada dos embeddings das referências: **257,59 ms**
- Memória total: não disponível pelo ambiente de medição

## Tempos reais

Todos os valores estão em milissegundos. O percentil 95 foi calculado sobre as 450 observações subsequentes.

| Configuração | Fase | n | Média | Mediana | Desvio-padrão | Mínimo | Máximo | P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Palavras-chave | primeira inferência | 1 | 0,0406 | 0,0406 | 0,0000 | 0,0406 | 0,0406 | n/d |
| Embeddings padrão | primeira inferência | 1 | 27,6974 | 27,6974 | 0,0000 | 27,6974 | 27,6974 | n/d |
| Embeddings rejeição (0,50) | primeira inferência | 1 | 15,6141 | 15,6141 | 0,0000 | 15,6141 | 15,6141 | n/d |
| Palavras-chave | subsequente | 450 | 0,0137 | 0,0133 | 0,0025 | 0,0102 | 0,0319 | 0,0160 |
| Embeddings padrão | subsequente | 450 | 15,0015 | 14,7290 | 2,6015 | 11,8235 | 45,9154 | 16,5027 |
| Embeddings rejeição (0,50) | subsequente | 450 | 15,6465 | 15,4358 | 1,6894 | 12,7924 | 30,3734 | 18,2229 |

A inicialização e a preparação de referências não foram incluídas na latência por pergunta. A gravação dos resultados e a impressão no terminal ocorreram fora dos intervalos cronometrados. O modo padrão e o modo de rejeição foram medidos após o mesmo modelo estar carregado; a diferença entre eles inclui somente a decisão da configuração existente, não um novo carregamento.

## Artefatos

- Execuções individuais: `results/semana5/latencia_execucoes.csv`
- Resumo estatístico: `results/semana5/latencia_resumo.csv`
- Ambiente: `results/semana5/ambiente_experimental.json`
- Gráfico: `results/semana5/grafico_latencia.svg`

As medições são uma execução observacional em CPU e não constituem uma estimativa universal de desempenho nem uma comparação estatística entre hardware.
