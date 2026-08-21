from fastapi import FastAPI, HTTPException

from app.predictor import predictor
from app.schemas import HealthResponse, NotificationPrediction, NotificationRequest

app = FastAPI(
    title="Classificador de Notificações",
    description="Classifica notificações jurídicas usando embeddings "
    "de Sentence Transformer "
    "e um classificador (Regressão Logística ou SVM).",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_ready else "modelo_nao_carregado",
        modelo_carregado=predictor.is_ready,
        classes=predictor.classes,
    )


@app.post("/classificar", response_model=NotificationPrediction)
def classificar(payload: NotificationRequest) -> NotificationPrediction:
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Rode o treinamento antes de usar a API "
            "(poetry run python -m scripts.train).",
        )

    try:
        result = predictor.predict(payload.sentenca)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return NotificationPrediction(**result)
