from __future__ import annotations

import json

import joblib

from scripts.embedder import (
    CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    MODELS_DIR,
    embed,
)
from scripts.priority import calculate_priority_score, get_priority_label


class Predictor:
    def __init__(self) -> None:
        self._classifier = None
        self._label_encoder = None
        self._models: dict[str, tuple[object, object]] = {}
        self._metadata: dict = {}
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        self._models = {}
        try:
            self._classifier = joblib.load(CLASSIFIER_PATH)
            self._label_encoder = joblib.load(LABEL_ENCODER_PATH)
            self._models["melhor"] = (self._classifier, self._label_encoder)
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

        for model_name, paths in self._metadata.get("models", {}).items():
            try:
                classifier = joblib.load(MODELS_DIR / paths["classifier"])
                label_encoder = joblib.load(MODELS_DIR / paths["label_encoder"])
            except (FileNotFoundError, KeyError):
                continue
            self._models[model_name] = (classifier, label_encoder)

        legacy_name = self._metadata.get("classifier")
        if "melhor" in self._models and legacy_name:
            self._models.setdefault(legacy_name, self._models["melhor"])

        if self._models and self._classifier is None:
            default_name = self._metadata.get("classifier")
            self._classifier, self._label_encoder = self._models.get(
                default_name, next(iter(self._models.values()))
            )

    def reload(self) -> None:
        """
        Recarrega os artefatos do modelo do disco (usado após um novo treino).
        """
        self._load()

    @property
    def is_ready(self) -> bool:
        return bool(self._models)

    @property
    def model_names(self) -> list[str]:
        return [name for name in self._models if name != "melhor"]

    @property
    def default_model(self) -> str | None:
        return self._metadata.get("classifier") or (
            self.model_names[0] if self.model_names else None
        )

    @property
    def classes(self) -> list[str]:
        if not self.is_ready:
            return []
        return list(self._label_encoder.classes_)

    @property
    def metadata(self) -> dict:
        return self._metadata

    def predict(self, sentenca: str, model_name: str | None = None) -> dict:
        if not self.is_ready:
            raise RuntimeError(self._load_error or "Modelo não carregado.")

        selected_name = model_name or self.default_model
        if selected_name not in self._models:
            raise ValueError(
                f"Modelo '{selected_name}' não encontrado. "
                f"Opções: {', '.join(self.model_names)}."
            )

        classifier, label_encoder = self._models[selected_name]

        embedding = embed([sentenca])
        probabilities = classifier.predict_proba(embedding)[0]

        class_indices = probabilities.argsort()[::-1]
        best_idx = class_indices[0]

        classe = label_encoder.inverse_transform([best_idx])[0]
        
        # Calcular Score de Prioridade
        priority_score = calculate_priority_score(sentenca)
        prioridade = get_priority_label(priority_score)

        return {
            "modelo": selected_name,
            "classe": classe,
            "prioridade": prioridade,
            "priority_score": round(priority_score, 2),
            "confianca": float(probabilities[best_idx]),
            "probabilidades": {
                label_encoder.inverse_transform([i])[0]: float(probabilities[i])
                for i in range(len(probabilities))
            },
        }


predictor = Predictor()
