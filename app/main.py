from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import training_state
from app.predictor import predictor
from app.schemas import (
    HealthResponse,
    ModelsResponse,
    NotificationPrediction,
    NotificationRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
)

app = FastAPI(
    title="Classificador de Notificações",
    description="Classifica notificações jurídicas usando embeddings "
    "de Sentence Transformer "
    "e um classificador (Regressão Logística ou SVM).",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_ready else "modelo_nao_carregado",
        modelo_carregado=predictor.is_ready,
        classes=predictor.classes,
        modelos=predictor.model_names,
    )


@app.get("/modelos", response_model=ModelsResponse)
def modelos() -> ModelsResponse:
    return ModelsResponse(
        modelos=predictor.model_names,
        modelo_padrao=predictor.default_model,
    )


@app.post(
    "/treinar",
    response_model=TrainingStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def treinar() -> TrainingStartResponse:
    """
    Dispara o pipeline de treinamento (embeddings + treino dos modelos
    candidatos + seleção e persistência do melhor) em background.
    Use GET /treinar/status para acompanhar o progresso.
    """
    iniciado = training_state.start_training()
    if not iniciado:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Já existe um treinamento em andamento.",
        )
    return TrainingStartResponse(
        status="em_andamento",
        mensagem="Treinamento iniciado. Acompanhe em GET /treinar/status.",
    )


@app.get("/treinar/status", response_model=TrainingStatusResponse)
def treinar_status() -> TrainingStatusResponse:
    return TrainingStatusResponse(**training_state.get_status())


@app.post("/classificar", response_model=NotificationPrediction)
def classificar(payload: NotificationRequest) -> NotificationPrediction:
    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Rode o treinamento antes de usar a API "
            "(poetry run python -m scripts.train).",
        )

    try:
        result = predictor.predict(payload.sentenca, payload.modelo)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return NotificationPrediction(**result)


STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
