from pydantic import BaseModel, Field


class NotificationRequest(BaseModel):
    sentenca: str = Field(
        ...,
        min_length=1,
        description="Texto da notificação a ser classificada.",
        examples=["Prazo para contestação encerra amanhã às 18h."],
    )


class NotificationPrediction(BaseModel):
    classe: str
    prioridade: str
    priority_score: float
    confianca: float
    probabilidades: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    classes: list[str]


class TrainingStartResponse(BaseModel):
    status: str
    mensagem: str


class TrainingStatusResponse(BaseModel):
    status: str = Field(
        ..., description="ocioso | em_andamento | concluido | erro"
    )
    mensagem: str
    melhor_modelo: str | None = None
    resultados: dict | None = None
    iniciado_em: str | None = None
    finalizado_em: str | None = None
