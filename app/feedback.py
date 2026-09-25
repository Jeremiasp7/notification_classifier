from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone

from scripts.loader import DATA_DIR, TRAIN_PATH

SENTENCES_DIR = DATA_DIR / "sentencas"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"
_write_lock = threading.Lock()


def known_classes() -> list[str]:
    """Return the canonical classes declared by the sentence corpus."""
    return sorted(path.stem for path in SENTENCES_DIR.glob("*.txt"))


def register_correction(
    sentenca: str,
    classe_predita: str,
    classe_correta: str,
) -> None:
    """Persist a human correction in the training set and audit log."""
    sentenca = sentenca.strip()
    classe_predita = classe_predita.strip()
    classe_correta = classe_correta.strip()

    if not sentenca:
        raise ValueError("A sentença não pode ser vazia.")
    if classe_correta not in known_classes():
        raise ValueError(f"Classe inválida: {classe_correta}.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    with _write_lock:
        _append_training_example(sentenca, classe_correta)
        _append_feedback_log(
            sentenca, classe_predita, classe_correta, timestamp
        )


def _append_training_example(sentenca: str, classe: str) -> None:
    fieldnames = ["id", "sentenca", "classe"]
    rows = []
    if TRAIN_PATH.exists():
        with TRAIN_PATH.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file, delimiter=";"))

    numeric_ids = []
    for row in rows:
        try:
            numeric_ids.append(int(row["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    next_id = max(numeric_ids, default=0) + 1

    with TRAIN_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        if TRAIN_PATH.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({"id": next_id, "sentenca": sentenca, "classe": classe})


def _append_feedback_log(
    sentenca: str,
    classe_predita: str,
    classe_correta: str,
    timestamp: str,
) -> None:
    fieldnames = [
        "timestamp",
        "sentenca",
        "classe_predita",
        "classe_correta",
    ]
    file_exists = FEEDBACK_PATH.exists() and FEEDBACK_PATH.stat().st_size > 0
    with FEEDBACK_PATH.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": timestamp,
                "sentenca": sentenca,
                "classe_predita": classe_predita,
                "classe_correta": classe_correta,
            }
        )