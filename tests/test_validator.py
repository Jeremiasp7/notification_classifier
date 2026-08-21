import pytest

from scripts.embedder import CLASSIFIER_PATH, LABEL_ENCODER_PATH
from scripts.validator import evaluate, load_artifacts

MODEL_TRAINED = CLASSIFIER_PATH.exists() and LABEL_ENCODER_PATH.exists()


def test_load_artifacts_raises_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.validator.CLASSIFIER_PATH", tmp_path / "nope.joblib")
    monkeypatch.setattr("scripts.validator.LABEL_ENCODER_PATH", tmp_path / "nope_le.joblib")
    with pytest.raises(FileNotFoundError):
        load_artifacts()


@pytest.mark.skipif(not MODEL_TRAINED, reason="'python -m scripts.train' antes deste teste")
def test_evaluate_runs_on_val_and_test(capsys):
    evaluate(split="val")
    evaluate(split="test")
    output = capsys.readouterr().out
    assert "Avaliação no conjunto de test" in output
