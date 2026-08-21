from __future__ import annotations

import argparse

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from scripts.embedder import CLASSIFIER_PATH, LABEL_ENCODER_PATH, embed
from scripts.loader import load_test, load_val

SPLIT_LOADERS = {
    "val": load_val,
    "test": load_test,
}


def load_artifacts():
    if not CLASSIFIER_PATH.exists() or not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Rode primeiro: poetry run python -m scripts.train"
        )
    classifier = joblib.load(CLASSIFIER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return classifier, label_encoder


def evaluate(split: str = "test") -> None:
    df = SPLIT_LOADERS[split]()
    classifier, label_encoder = load_artifacts()

    X = embed(df["sentenca"].tolist())
    y_true = label_encoder.transform(df["classe"])
    y_pred = classifier.predict(X)

    print(f"Avaliação no conjunto de {split} ({len(df)} exemplos)\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )
    print("Matriz de confusão (linhas=real, colunas=previsto):")
    print(label_encoder.classes_)
    print(confusion_matrix(y_true, y_pred))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia o classificador salvo.")
    parser.add_argument("--split", choices=list(SPLIT_LOADERS.keys()), default="test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(split=args.split)
