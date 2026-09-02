from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    MLP classifier for sentence embeddings with tensorflow.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_size: int = 128,
        epochs: int = 150,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.network: tf.keras.Model | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NeuralNetworkClassifier:
        features = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("X deve ser 2D e ter o mesmo número de linhas que y.")

        tf.keras.utils.set_random_seed(self.random_state)
        self.classes_ = np.unique(labels)
        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(features.shape[1],)),
                tf.keras.layers.Dense(self.hidden_size, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(len(self.classes_), activation="softmax"),
            ]
        )
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            ),
            loss="sparse_categorical_crossentropy",
        )
        class_counts = np.bincount(labels, minlength=len(self.classes_))
        class_weights = len(labels) / (len(self.classes_) * np.maximum(class_counts, 1))
        self.network.fit(
            features,
            labels,
            epochs=self.epochs,
            batch_size=min(32, len(features)),
            class_weight=dict(enumerate(class_weights.tolist())),
            verbose=0,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("A rede neural ainda não foi treinada.")
        features = np.asarray(X, dtype=np.float32)
        return np.asarray(self.network.predict(features, verbose=0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return probabilities.argmax(axis=1)
