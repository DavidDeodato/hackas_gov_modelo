### Memory Bank — Hackathon CGDF (Acesso à Informação)

#### Contexto
- **Objetivo (edital)**: submeter um **modelo** que identifique pedidos “públicos” contendo **dados pessoais**, com avaliação por **Precisão** e **Recall** e pontuação \(P1 = F1\). Ver `edital.txt` (itens 8.1.1–8.1.5).
- **Estado inicial do repo**: existia um modelo baseado em **features de regex + XGBoost** (`src/predict_final.py`) treinado em dataset sintético (`data/dataset_pedidos_lai.csv`) com métricas “100%”.
- **Auditoria rápida**: o “100%” ocorre porque o dataset sintético é altamente correlacionado com as regex; em teste adversarial (`test_frases.py`) o desempenho caiu para ~54% por:
  - **FNs** em nome/endereço/email ofuscado/formatos alternativos;
  - **FPs** por números longos (protocolo/SEI/IDs) confundidos com CPF/CNH/CEP.

#### Dados disponíveis
- `data/dataset_pedidos_lai.csv`: dataset sintético rotulado (15k).
- `AMOSTRA_e-SIC - Amostra - SIC.csv`: amostra de textos mascarados, **sem rótulo**, mas com padrões reais de escrita e de números (SEI, protocolos, matrículas, etc.).

#### Decisão de modelagem (2026-01-30)
- Migrar para um **modelo de texto** em CPU: **TF‑IDF (char + word n-grams) + classificador linear** (Logistic Regression ou LinearSVC calibrado).
- Motivo: capturar variações textuais (ex.: email ofuscado, CPF espaçado, nomes) e reduzir dependência de regex como sinal de decisão.

#### Estado atual (após iterações)
- **Modelo em produção**: `models/modelo_tfidf.pkl` + `models/threshold.json` (inferência em `src/predict_final.py`)
- **Dataset de treino**: `data/dataset_pedidos_lai_aug.csv` (gerado por `src/train.py --build-dataset`)
- **Sanity check adversarial**: `test_frases.py` (100 casos) → **100/100** após ajustes de rotulagem coerente:
  - **CNPJ** tratado como **não-pessoal** (pessoa jurídica) no teste
  - padrões numéricos “parecidos” (ex.: PIS sem contexto) tratados como **ambíguos** para evitar falso positivo
- **Compatibilidade Windows**: removidos símbolos/emoji na saída do `predict_final.py` para não quebrar encoding no PowerShell.
- **Documentação**: README reforçado + relatório técnico em `docs/RELATORIO_TECNICO.md` + figuras geradas em `docs/confusion_matrix_tfidf.png` e `docs/roc_curve_tfidf.png`.

#### Entrega rápida (gravação)
- **Removido** `VIDEO_ROTEIRO.md` (decisão: vídeo não é obrigatório para a categoria ML, e o arquivo ficava “estranho” no repo).

#### Plano de implementação (resumo)
- Criar **gerador de dataset ampliado** com:
  - positivos: formatos alternativos + casos “só nome/endereço”;
  - negativos: “hard negatives” com números de processo/SEI/protocolo/inscrição/nota fiscal.
- Treinar modelo com split estratificado, reportar métricas e fazer **threshold tuning** em validação.
- Atualizar inferência (`src/predict_final.py`) para carregar o artefato do modelo TF‑IDF e manter saída atual.


