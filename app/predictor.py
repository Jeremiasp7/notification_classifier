from __future__ import annotations

import joblib

from scripts.embedder import CLASSIFIER_PATH, LABEL_ENCODER_PATH, embed


class Predictor:
    def __init__(self) -> None:
        self._classifier = None
        self._label_encoder = None
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            self._classifier = joblib.load(CLASSIFIER_PATH)
            self._label_encoder = joblib.load(LABEL_ENCODER_PATH)
        except FileNotFoundError:
            self._load_error = (
                "Modelo não encontrado. Rode 'poetry run python -m scripts.train' "
                "para treinar e gerar os artefatos em /models."
            )

    @property
    def is_ready(self) -> bool:
        return self._classifier is not None and self._label_encoder is not None

    @property
    def classes(self) -> list[str]:
        if not self.is_ready:
            return []
        return list(self._label_encoder.classes_)

    def predict(self, sentenca: str) -> dict:
        if not self.is_ready:
            raise RuntimeError(self._load_error or "Modelo não carregado.")

        embedding = embed([sentenca])
        probabilities = self._classifier.predict_proba(embedding)[0]

        class_indices = probabilities.argsort()[::-1]
        best_idx = class_indices[0]

        return {
            "classe": self._label_encoder.inverse_transform([best_idx])[0],
            "confianca": float(probabilities[best_idx]),
            "probabilidades": {
                self._label_encoder.inverse_transform([i])[0]: float(probabilities[i])
                for i in range(len(probabilities))
            },
        }


predictor = Predictor()
