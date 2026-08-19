from pathlib import Path

import pandas as pd


class DatasetLoader:
    """
    Class responsable for the dataset loading of train, validation and test
    """

    TRAIN_DATABASE = "database_notification_law_train.csv"
    VALIDATION_DATABASE = "database_notification_law_val.csv"
    TEST_DATABASE = "database_notification_law_test.csv"

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                "Expect: .csv"
            )

        return pd.read_csv(path)

    def load_train_dataset(self) -> pd.DataFrame:
        return self.load(self.TRAIN_DATABASE)

    def load_validation_dataset(self) -> pd.DataFrame:
        return self.load(self.VALIDATION_DATABASE)

    def load_test_dataset(self) -> pd.DataFrame:
        return self.load(self.TEST_DATABASE)

    def load_all_datasets(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.load_train(),
            "validation": self.load_validation(),
            "test": self.load_test(),
        }
