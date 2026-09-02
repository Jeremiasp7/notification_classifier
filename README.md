# notification_classifier

Classificador de notificações jurídicas: **Sentence Transformer** (embeddings) +
**Regressão Logística / SVM / XGBoost / Rede Neural Tensorflow** (classificação), servido via **FastAPI**. Sem banco de
dados — fluxo direto: request → embedding → classificador → response.

## Estrutura

```
app/
  data/            # CSVs de treino/validação/teste
  main.py          # FastAPI app (endpoints)
  predictor.py     # carrega o modelo e faz predição
  schemas.py       # request/response (Pydantic)
models/            # artefatos treinados (gerados pelo train.py, não versionados)
scripts/
  embedder.py      # config central do Sentence Transformer + caminhos dos artefatos
  loader.py        # carrega os CSVs
  train.py         # treina, compara e salva o melhor modelo
  validator.py     # avalia o modelo salvo em val/test
tests/
```

## Instalação

```bash
poetry install
```

## 1. Gerar datasets

A partir das sentenças, cria um dataset unificado e depois divide esse dataset em três:
treino, validação e teste (70/15/15)

```bash
poetry run python -m scripts.generate_dataset
poetry run python -m scripts.split_dataset
```

## 2. Treinar

Treina Regressão Logística, SVM, XGBoost e uma rede neural PyTorch. Avalia os quatro
na validação e salva apenas o melhor:

```bash
poetry run python -m scripts.train --plot 
```

Isso gera em `models/`: `classifier.joblib`, `label_encoder.joblib`, `metadata.json`.
Com `--plot`, os relatórios e matrizes de validação de cada modelo são salvos em
`reports/`, junto com o gráfico de comparação.

## 3. Validar / Testar

```bash
poetry run python -m scripts.validator --split val --plot
poetry run python -m scripts.validator --split test --plot --show
```

Mostra `classification_report` (precision/recall/f1 por classe) e matriz de confusão.

## 4. Subir a API

```bash
poetry run uvicorn app.main:app --reload
```

- `GET  /health` — status do modelo e classes disponíveis
- `POST /classificar` — `{"sentenca": "..."}` → `{"classe", "confianca", "probabilidades"}`

Exemplo:

```bash
curl -X POST http://127.0.0.1:8000/classificar \
  -H "Content-Type: application/json" \
  -d '{"sentenca": "Audiência de conciliação marcada para sexta-feira."}'
```

## 5. Testes

```bash
poetry run pytest
```

(os testes de API/validador que dependem do modelo treinado são pulados
automaticamente se `scripts/train.py` ainda não tiver sido executado)

## Notas de projeto

- **Modelo de embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` — multilíngue,
  leve e roda bem em CPU; boa cobertura para português. Trocar é uma linha só
  (`MODEL_NAME` em `scripts/embedder.py`).
- **Escolha do classificador**: `train.py` treina os quatro candidatos e escolhe
  automaticamente o de maior F1-macro na validação. A rede neural é um MLP pequeno
  em TensorFlow, adequado aos embeddings já normalizados e ao tamanho do dataset.
- **Sem banco de dados**: os embeddings são gerados on-the-fly a cada request; o
  único estado persistido em disco são os artefatos do classificador (`models/`),
  carregados uma única vez na subida da API.
