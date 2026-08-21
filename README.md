# notification_classifier

Classificador de notificações jurídicas: **Sentence Transformer** (embeddings) +
**Regressão Logística / SVM** (classificação), servido via **FastAPI**. Sem banco de
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

## 1. Treinar

Treina Regressão Logística e SVM, avalia os dois na validação e salva o melhor:

```bash
poetry run python -m scripts.train --plot 
```

Isso gera em `models/`: `classifier.joblib`, `label_encoder.joblib`, `metadata.json`.

## 2. Validar / Testar

```bash
poetry run python -m scripts.validator --split val --plot
poetry run python -m scripts.validator --split test --plot --show
```

Mostra `classification_report` (precision/recall/f1 por classe) e matriz de confusão.

## 3. Subir a API

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

## 4. Testes

```bash
poetry run pytest
```

(os testes de API/validador que dependem do modelo treinado são pulados
automaticamente se `scripts/train.py` ainda não tiver sido executado)

## Notas de projeto

- **Modelo de embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` — multilíngue,
  leve e roda bem em CPU; boa cobertura para português. Trocar é uma linha só
  (`MODEL_NAME` em `scripts/embedder.py`).
- **Escolha do classificador**: `train.py` treina Regressão Logística e SVM linear
  e escolhe automaticamente o de maior F1-macro na validação (dataset é pequeno e
  balanceado, então os dois são baratos de treinar).
- **Sem banco de dados**: os embeddings são gerados on-the-fly a cada request; o
  único estado persistido em disco são os artefatos do classificador (`models/`),
  carregados uma única vez na subida da API.
