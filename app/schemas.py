from pydantic import BaseModel, Field


class NotificationRequest(BaseModel):
    sentenca: str = Field(
        ...,
        min_length=1,
        description="Texto da notificação a ser classificada.",
        examples=["Prazo para contestação encerra amanhã às 18h."],
    )
    modelo: str | None = Field(
        default=None,
        description="Modelo salvo que deve fazer a classificação. Se omitido, usa o melhor modelo.",
    )


class NotificationPrediction(BaseModel):
    modelo: str
    classe: str
    prioridade: str
    priority_score: float
    confianca: float
    probabilidades: dict[str, float]


class FeedbackRequest(BaseModel):
    sentenca: str = Field(..., min_length=1)
    classe_predita: str = Field(..., min_length=1)
    classe_correta: str = Field(..., min_length=1)


class FeedbackResponse(BaseModel):
    status: str
    mensagem: str


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    classes: list[str]
    modelos: list[str]


class ModelsResponse(BaseModel):
    modelos: list[str]
    modelo_padrao: str | None = None


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
