# Relatório de reprodutibilidade — Semana 5

**Status:** **reprodução concluída**.
**Identificador da execução:** `reproducao_20261025_221810`.

A procedência dos arquivos existentes em `reexecucao_semana2/` foi registrada por hash, mas sua existência isolada não foi tratada como prova de execução concluída.

## Execução verificável

- Comando: `PowerShell $env:HF_HUB_OFFLINE=1 $env:TRANSFORMERS_OFFLINE=1 C:\Users\Kleber\Desktop\TCC\Repositório\.venv\Scripts\python.exe evaluate.py --modo completo --resultados results/semana5/reproducao_final --sem-graficos`
- Código de saída: **0**
- Diretório: `C:\Users\Kleber\Desktop\TCC\Repositório\results\semana5\reproducao_final`
- Commit observado: `6a6edacdfdfed8da804e776b06a0b6ad48d2e0d0`
- Duração: **11.586 s**
- Modelo: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Python: `3.14.7`
- Partições esperadas: validação 40 perguntas, teste 60 perguntas, com seleção do threshold somente na validação.

## Arquivos

- Obrigatórios ausentes: `[]`.
- Arquivos gerados: **30**; hashes no `manifesto_execucao.json`.

## Comparação com o histórico

| Método | Situação | Histórico | Reprodução |
|---|---|---|---|
| Embeddings com threshold 0.50 | coincidente | {'metodo': 'Embeddings com threshold 0.50', 'acuracia': np.float64(0.7666666666666667), 'macro_f1': np.float64(0.766308462008793)} | {'metodo': 'Embeddings com threshold 0.50', 'acuracia': np.float64(0.7666666666666667), 'macro_f1': np.float64(0.766308462008793)} |
| Embeddings sem threshold | coincidente | {'metodo': 'Embeddings sem threshold', 'acuracia': np.float64(0.6666666666666666), 'macro_f1': np.float64(0.671173030615755)} | {'metodo': 'Embeddings sem threshold', 'acuracia': np.float64(0.6666666666666666), 'macro_f1': np.float64(0.671173030615755)} |
| Palavras-chave | coincidente | {'metodo': 'Palavras-chave', 'acuracia': np.float64(0.5666666666666667), 'macro_f1': np.float64(0.5904431610700961)} | {'metodo': 'Palavras-chave', 'acuracia': np.float64(0.5666666666666667), 'macro_f1': np.float64(0.5904431610700961)} |

Nenhum histórico foi sobrescrito. Divergências, se presentes, devem ser investigadas por versões, partições, modelo e implementação.
