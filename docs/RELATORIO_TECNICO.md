### Relatório Técnico — Categoria Acesso à Informação (Hackathon CGDF)

#### 1) Resumo executivo
Este trabalho entrega um **modelo executável** que, dado o texto de um pedido LAI marcado como público, estima a probabilidade de conter **dados pessoais de pessoa natural** (ex.: CPF, RG, telefone, e-mail, endereço, nome) e, portanto, indicar necessidade de reclassificação como não público.

Entregáveis principais no repositório:
- modelo serializado (`models/modelo_tfidf.pkl`) + threshold (`models/threshold.json`)
- script de inferência (`src/predict_final.py`)
- pipeline reprodutível de geração de dados e treino (`src/train.py`)
- métricas exportadas (`models/metricas_tfidf.json`)
- figuras de avaliação (`docs/*.png`)

#### 2) Como a CGDF avalia (P1)
Segundo o edital, a pontuação técnica P1 é baseada em **Precisão** e **Recall** e usa a fórmula:

- Precisão = \(VP / (VP + FP)\)
- Recall = \(VP / (VP + FN)\)
- P1 = \(2 \times (Precisão \times Recall) / (Precisão + Recall)\) (equivalente ao F1)

Arquivo de referência: `edital.txt` (itens 8.1.5.2.1, 8.1.5.2.2, 8.1.5.2.3).

#### 3) Procedência dos dados e como foram obtidos
Este repositório trabalha com **dados sintéticos** e **amostras mascaradas**, alinhado ao edital (nenhum dado pessoal real é distribuído aos participantes).

Fontes presentes no repositório:
- `data/dataset_pedidos_lai.csv` (base sintética rotulada)
  - já existia no repositório como ponto de partida (textos + rótulo binário `contem_dados_pessoais`)
- `AMOSTRA_e-SIC - Amostra - SIC.csv` (amostra mascarada, sem rótulo)
  - usada para capturar linguagem realista e padrões de numeração comuns (SEI, protocolos, matrículas, ocorrências, nota fiscal, etc.)
  - não é usada diretamente como rótulo; é usada para criar **hard negatives** após limpeza

Dataset gerado durante o trabalho:
- `data/dataset_pedidos_lai_aug.csv` (dataset ampliado rotulado)
  - gerado por `src/train.py --build-dataset`
  - objetivo: reduzir overfitting em “regex fáceis” e melhorar generalização para o subconjunto de controle

#### 4) Como o dataset sintético ampliado é gerado (auditável)
O gerador em `src/train.py` cria exemplos a partir de templates e regras determinísticas:

- Exemplos positivos (classe 1):
  - incluem formatos variados e realistas, por exemplo:
    - “me chamo Fulano…” (nome sem números)
    - CPF formatado e CPF espaçado
    - telefone com/sem parênteses
    - e-mail normal e e-mail ofuscado (“[arroba]”, “(at)”, “ at ”)
  - CPFs formatados são gerados por função determinística com dígitos verificadores (checksum) para evitar padrões irreais

- Exemplos negativos (classe 0):
  - incluem “pedidos LAI típicos” e, principalmente, **hard negatives** com números que parecem documentos:
    - SEI, protocolo, matrícula, ocorrência, nota fiscal, inscrição, empenho, etc.
  - também incorpora textos da amostra mascarada como base de linguagem, após aplicar *scrub* de padrões óbvios (ver `src/features_pii.py::scrub_pii`)

Essa etapa é o principal mecanismo de robustez contra falsos positivos em números longos e contra falsos negativos em nomes/endereço sem formato rígido.

Detalhes implementacionais (para auditoria):
- **Aleatoriedade controlada**: `seed=42` (ver `src/train.py`).
- **Volume padrão do gerador**: `extra_pos=8000`, `extra_neg=8000` (ajustável por CLI).
- **CPF válido**: gerado por `gerar_cpf_valido()` usando cálculo de dígito verificador (evita exemplos “CPF impossível”).
- **E-mail ofuscado**: gerado por `gerar_email_ofuscado()` (ex.: `joao [arroba] dominio.com`, `joao (at) dominio.com`, `joao at dominio.com`).
- **Hard negatives**: gerados em `_negativos_hard()` com padrões explícitos (SEI, ocorrência, matrícula, empenho, nota fiscal) e também com textos mascarados do e-SIC após `scrub_pii()`.

Exemplos concretos de templates (extraídos do gerador):
- Positivo: `\"Me chamo {NOME}. Solicito acesso aos autos do meu processo SEI.\"`
- Positivo: `\"CPF: {CPF_ESPACADO}. Solicito informações.\"`
- Negativo: `\"Solicito acesso integral aos autos do Processo SEI {SEI}, com urgência.\"`
- Negativo: `\"Nota fiscal: {PROTO11}. Solicito dados consolidados.\"`

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

Parâmetros relevantes (ver `src/train.py`):
- split estratificado 70/15/15 com seed fixa (reprodutível)
- ajuste de threshold por busca em grade para maximizar F1 na validação

Configuração usada (treino padrão):
- TF‑IDF palavras: `ngram_range=(1,2)`, `min_df=2`, `max_df=0.98`, `strip_accents=\"unicode\"`
- TF‑IDF caracteres: `analyzer=\"char_wb\"`, `ngram_range=(3,5)`, `min_df=2`, `strip_accents=\"unicode\"`
- Logistic Regression: `solver=\"saga\"`, `max_iter=3000`, `class_weight=\"balanced\"`, `random_state=42`

Motivo da escolha:
- n-grams de caracteres capturam formatos e variações (documentos com pontuação, espaços, e-mails ofuscados).
- n-grams de palavras capturam contexto (“cpf”, “meu nome”, “endereço”, “SEI”, “protocolo”, “nota fiscal”).
- threshold tuning permite controlar o trade-off **FP vs FN** diretamente na métrica do edital (F1).

#### 6) Reprodutibilidade (como reproduzir)
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
 - Curva Precision-Recall: `docs/pr_curve_tfidf.png`
 - Distribuição das probabilidades por classe: `docs/proba_hist_tfidf.png`
 - Top termos (coeficientes do modelo): `docs/top_features_tfidf.png`

Para regenerar:

```bash
py .\src\report_plots.py
```

#### 7.1) Diagnóstico na amostra e-SIC (sem rótulo)
A amostra `AMOSTRA_e-SIC - Amostra - SIC.csv` não possui coluna de rótulo. Portanto, não é possível calcular Precisão/Recall/F1 diretamente nela.

O script abaixo gera um relatório objetivo do comportamento do modelo nessa amostra (percentual de positivos preditos, histograma de probabilidades e exemplos extremos), além de exportar um CSV com as predições:

```bash
py .\src\evaluate_amostra.py
```

Saídas:
- `docs/amostra_predictions.csv`
- `docs/amostra_relatorio.md`
- `docs/amostra_proba_hist.png`

Notas sobre interpretação:
- Curva ROC pode ser alta mesmo quando o threshold está mal calibrado; por isso, mantemos também PR curve e histograma de probabilidades.
- `top_features_tfidf.png` mostra termos mais associados a classe 1 e a classe 0 (coeficientes do modelo), ajudando auditoria do comportamento.

#### 8) Explicabilidade (tipos detectados)
Além da classe binária, o script retorna `tipos_dados_encontrados` usando detecção explicável em `src/features_pii.py`.

Isso **não** é usado como decisão final do classificador; serve para:
- melhorar confiança/interpretabilidade
- facilitar auditoria do output

#### 9) Limitações e riscos
- O dataset rotulado disponível no repo é sintético; por isso, mitigamos risco de overfitting adicionando hard negatives inspirados em amostras reais.
- Alguns padrões numéricos são inerentemente ambíguos fora de contexto (ex.: “11 dígitos” pode ser várias coisas). O classificador textual tende a resolver isso via contexto semântico (palavras ao redor), e o gerador inclui exemplos negativos desse tipo.

