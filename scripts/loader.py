from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"

REQUIRED_COLUMNS = {"id", "sentenca", "classe"}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path, sep=';')

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing}")

    df = df.dropna(subset=["sentenca", "classe"]).reset_index(drop=True)
    df["sentenca"] = df["sentenca"].astype(str).str.strip()
    df["classe"] = df["classe"].astype(str).str.strip()

    return df


def load_train() -> pd.DataFrame:
    return _load_csv(TRAIN_PATH)


def load_val() -> pd.DataFrame:
    return _load_csv(VAL_PATH)


def load_test() -> pd.DataFrame:
    return _load_csv(TEST_PATH)


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "train": load_train(),
        "val": load_val(),
        "test": load_test(),
    }


if __name__ == "__main__":
    for split, df in load_all().items():
        print(f"{split}: {len(df)} linhas | classes: {sorted(df['classe'].unique())}")
