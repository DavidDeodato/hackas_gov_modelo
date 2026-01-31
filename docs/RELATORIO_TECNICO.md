### Relatório Técnico — Categoria Acesso à Informação (Hackathon CGDF)

#### 1) Resumo executivo
Objetivo: identificar automaticamente pedidos de acesso à informação marcados como “públicos” que contêm **dados pessoais** e, portanto, deveriam ser classificados como **não públicos**.

Entregável principal: **modelo executável** + documentação reprodutível (treino e inferência).

#### 2) Como a CGDF avalia (P1)
Segundo o edital, a pontuação técnica P1 é baseada em **Precisão** e **Recall** e usa a fórmula:

- Precisão = \(VP / (VP + FP)\)
- Recall = \(VP / (VP + FN)\)
- P1 = \(2 \times (Precisão \times Recall) / (Precisão + Recall)\) (equivalente ao F1)

Arquivo de referência: `edital.txt` (itens 8.1.5.2.1, 8.1.5.2.2, 8.1.5.2.3).

#### 3) Dados (o que existe no repositório)
- **Base rotulada (sintética)**: `data/dataset_pedidos_lai.csv`
  - usada como ponto de partida
- **Amostra real (sem rótulo)**: `AMOSTRA_e-SIC - Amostra - SIC.csv`
  - usada como fonte de *linguagem* e, principalmente, de **hard negatives** (padrões de SEI, protocolo, matrícula, ocorrências, nota fiscal etc.)
  - passa por *scrub* de padrões óbvios para evitar vazamento ao usar como template
- **Dataset ampliado (gerado)**: `data/dataset_pedidos_lai_aug.csv`
  - gerado por `src/train.py --build-dataset`

#### 4) Motivação da abordagem
O baseline anterior (regex + boosting) tende a:
- **errar por FN** quando há dados pessoais sem um formato numérico rígido (ex.: “me chamo Fulano…”)
- **errar por FP** quando há números longos que parecem CPF/CEP (ex.: protocolo/SEI/nota fiscal)

Para capturar variações reais de escrita e formato, usamos um modelo textual com **TF‑IDF** em nível de palavra e de caracteres.

#### 5) Modelo
Pipeline (em CPU):
- `TfidfVectorizer` de palavras (n-grams 1–2)
- `TfidfVectorizer` de caracteres (char_wb 3–5)
- `LogisticRegression` (class_weight=balanced)
- seleção de **threshold** para maximizar F1 em validação

Artefatos exportados:
- `models/modelo_tfidf.pkl`
- `models/threshold.json`
- `models/metricas_tfidf.json`

#### 6) Reprodutibilidade (o que a banca roda)
1) Instalar deps:

```bash
pip install -r requirements.txt
```

2) Gerar dataset ampliado:

```bash
py .\src\train.py --build-dataset
```

3) Treinar e exportar:

```bash
py .\src\train.py --train
```

4) Inferência (CLI):

```bash
py .\src\predict_final.py "Meu CPF é 123.456.789-01. Solicito informações." --json
```

5) Sanity check adversarial:

```bash
py .\test_frases.py
```

#### 7) Métricas registradas no repositório
As métricas do último treino estão em `models/metricas_tfidf.json`.

Pontos importantes:
- As métricas locais são obtidas em split de teste do dataset **sintético ampliado**.
- A avaliação oficial é em um **subconjunto de controle** da CGDF (oculto ao participante), então a estratégia prioriza generalização (hard negatives + variações de escrita).

Figuras de avaliação (geradas a partir do modelo exportado):

- Matriz de confusão: `docs/confusion_matrix_tfidf.png`
- Curva ROC: `docs/roc_curve_tfidf.png`

Para regenerar:

```bash
py .\src\report_plots.py
```

#### 8) Explicabilidade (tipos detectados)
Além da classe binária, o script retorna `tipos_dados_encontrados` usando detecção explicável em `src/features_pii.py`.

Isso **não** é usado como decisão final do classificador; serve para:
- melhorar confiança/interpretabilidade
- facilitar auditoria do output

#### 9) Limitações e riscos
- O dataset rotulado disponível no repo é sintético; por isso, mitigamos risco de overfitting adicionando hard negatives inspirados em amostras reais.
- Alguns padrões numéricos são inerentemente ambíguos fora de contexto (ex.: “11 dígitos” pode ser várias coisas). O classificador textual tende a resolver isso via contexto semântico (palavras ao redor), e o gerador inclui exemplos negativos desse tipo.

