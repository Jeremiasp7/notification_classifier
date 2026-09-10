from __future__ import annotations

import json

import joblib

from scripts.embedder import (
    CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    embed,
)
from scripts.priority import calculate_priority_score, get_priority_label


class Predictor:
    def __init__(self) -> None:
        self._classifier = None
        self._label_encoder = None
        self._metadata: dict = {}
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            self._classifier = joblib.load(CLASSIFIER_PATH)
            self._label_encoder = joblib.load(LABEL_ENCODER_PATH)
            self._load_error = None
        except FileNotFoundError:
            self._classifier = None
            self._label_encoder = None
            self._load_error = (
                "Modelo não encontrado. Rode 'poetry run python -m scripts.train' "
                "(ou use o botão 'Treinar modelo') para gerar os artefatos em /models."
            )

        try:
            self._metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._metadata = {}

    def reload(self) -> None:
        """
        Recarrega os artefatos do modelo do disco (usado após um novo treino).
        """
        self._load()

    @property
    def is_ready(self) -> bool:
        return self._classifier is not None and self._label_encoder is not None

    @property
    def classes(self) -> list[str]:
        if not self.is_ready:
            return []
        return list(self._label_encoder.classes_)

    @property
    def metadata(self) -> dict:
        return self._metadata

    def predict(self, sentenca: str) -> dict:
        if not self.is_ready:
            raise RuntimeError(self._load_error or "Modelo não carregado.")

        embedding = embed([sentenca])
        probabilities = self._classifier.predict_proba(embedding)[0]

        class_indices = probabilities.argsort()[::-1]
        best_idx = class_indices[0]

        classe = self._label_encoder.inverse_transform([best_idx])[0]
        
        # Calcular Score de Prioridade
        priority_score = calculate_priority_score(sentenca)
        prioridade = get_priority_label(priority_score)

        return {
            "classe": classe,
            "prioridade": prioridade,
            "priority_score": round(priority_score, 2),
            "confianca": float(probabilities[best_idx]),
            "probabilidades": {
                self._label_encoder.inverse_transform([i])[0]: float(probabilities[i])
                for i in range(len(probabilities))
            },
        }


predictor = Predictor()
