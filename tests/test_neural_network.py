from io import BytesIO

import joblib
import numpy as np

from scripts.neural_network import NeuralNetworkClassifier


def test_neural_network_predicts_probabilities_and_survives_serialization():
    features = np.array(
        [
            [-1.0, -0.8],
            [-0.9, -1.1],
            [0.0, 0.1],
            [0.1, -0.1],
            [1.0, 0.8],
            [0.9, 1.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])

    model = NeuralNetworkClassifier(epochs=20).fit(features, labels)
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored_model = joblib.load(buffer)

    probabilities = restored_model.predict_proba(features)

    assert probabilities.shape == (len(features), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(restored_model.predict(features)) <= {0, 1, 2}
