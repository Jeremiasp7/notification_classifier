from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from scripts.embedder import (
    CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    MODEL_NAME,
    MODELS_DIR,
    embed,
)
from scripts.loader import load_train, load_val
from scripts.plots import plot_model_comparison

CANDIDATES = {
    "logreg": lambda: LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"),
    "svm": lambda: SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced"),
    "xgboost": lambda: XGBClassifier(eval_metric="mlogloss", random_state=42)
}


def train_and_select(model_choice: str | None = None, plot: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train()
    val_df = load_val()

    print(f"Treino: {len(train_df)} exemplos | Validação: {len(val_df)} exemplos")
    print(f"Gerando embeddings com '{MODEL_NAME}'...")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["classe"])
    y_val = label_encoder.transform(val_df["classe"])

    X_train = embed(train_df["sentenca"].tolist())
    X_val = embed(val_df["sentenca"].tolist())

    candidates = CANDIDATES if model_choice is None else {model_choice: CANDIDATES[model_choice]}

    best_name, best_model, best_f1 = None, None, -1.0
    results = {}

    for name, build_model in candidates.items():
        model = build_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        results[name] = {"accuracy": acc, "f1_macro": f1}

        print(f"  [{name}] accuracy={acc:.4f}  f1_macro={f1:.4f}")

        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    print(f"\nMelhor modelo: {best_name} (f1_macro={best_f1:.4f})")

    if plot and len(results) > 1:
        chart_path = plot_model_comparison(results)
        print(f"Gráfico de comparação salvo em: {chart_path}")

    joblib.dump(best_model, CLASSIFIER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    metadata = {
        "sentence_transformer": MODEL_NAME,
        "classifier": best_name,
        "classes": label_encoder.classes_.tolist(),
        "val_results": results,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Artefatos salvos em: {MODELS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o classificador de notificações.")
    parser.add_argument(
        "--model",
        choices=list(CANDIDATES.keys()),
        default=None,
        help="Força o uso de um modelo específico (logreg ou svm). "
        "Se omitido, treina os dois e escolhe o melhor via validação.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Gera gráfico comparando accuracy/f1_macro dos candidatos, salvo em /reports.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_select(model_choice=args.model, plot=args.plot)
