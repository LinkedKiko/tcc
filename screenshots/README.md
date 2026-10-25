# Capturas reais da Semana 4

Este diretório contém capturas reais da aplicação Streamlit, obtidas durante a execução do puzzle. As imagens são evidências visuais da interface e não foram geradas artificialmente.

## Inventário

| Arquivo | Estado ou conteúdo registrado |
|---|---|
| `01-inicial.png` | Tela inicial da sala |
| `02-relogio.png` | Relógio examinado |
| `03-cofre-fechado.png` | Cofre fechado |
| `04-cofre-aberto.png` | Cofre aberto |
| `05-fotografia.png` | Fotografia encontrada/disponível |
| `06-mural-completo.png` | Mural completo |
| `07-enciclopedia.png` | Enciclopédia consultada |
| `08-simbolos.png` | Símbolos da parede |
| `09-dica.png` | Captura de dica contextual do cofre |
| `10-vitoria.png` | Tela de vitória |
| `11-reset.png` | Estado após reiniciar a partida |

A captura `09-dica.png` documenta a apresentação de uma dica contextual do cofre. Para a descrição textual completa da política de subintenções, consulte `docs/correcoes_semana4.md`.

## Reprodução

Para reproduzir as evidências:

```powershell
python -m streamlit run streamlit_app.py
```

O progresso é mantido na sessão do navegador. As capturas registram o estado visual no momento de cada ação; elas não substituem os testes automatizados nem a validação da classificação com o modelo real.
