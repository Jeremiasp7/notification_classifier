from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationResult:
    """
    Shows the result of a dataset validation
    """

    valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self) -> bool:
        return self.valid


class DatasetValidator:
    """
    Class responsable for the validate of datasets
    """

    REQUIRED_COLUMNS = {"text", "label"}

    def validate_datasets(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        self._validate_columns(df, errors)
        self._validate_missing_values(df, errors)
        self._validate_empty_texts(df, errors)
        self._validate_labels(df, errors)
        self._validate_duplicates(df, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_splits(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> ValidationResult:
        """
        Validate the three datasets and verify if has a leak between them
        """

        errors: list[str] = []
        warnings: list[str] = []

        datasets = {
            "train": train,
            "validation": validation,
            "test": test,
        }

        for name, df in datasets.items():
            result = self.validate(df)

            errors.extend(
                f"[{name}] {error}"
                for error in result.errors
            )

            warnings.extend(
                f"[{name}] {warning}"
                for warning in result.warnings
            )

        self._validate_cross_split_duplicates(train, validation, test, errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_columns(self, df: pd.DataFrame, errors: list[str]) -> None:
        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            errors.append(
                f"Colunas obrigatórias ausentes: "
                f"{sorted(missing_columns)}"
            )

    def _validate_missing_values(self, df: pd.DataFrame, errors: list[str]) -> None:
        if "text" not in df.columns or "label" not in df.columns:
            return

        missing_text = df["text"].isna().sum()
        missing_label = df["label"].isna().sum()

        if missing_text > 0:
            errors.append(
                f"{missing_text} text have null values."
            )

        if missing_label > 0:
            errors.append(
                f"{missing_label} labels have null values."
            )

    def _validate_empty_texts(self, df: pd.DataFrame, errors: list[str]) -> None:
        if "text" not in df.columns:
            return

        empty_texts = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if empty_texts > 0:
            errors.append(
                f"{empty_texts} texts are empty"
            )

    def _validate_labels(
        self,
        df: pd.DataFrame,
        errors: list[str],
    ) -> None:
        if "label" not in df.columns:
            return

        labels = df["label"].dropna()

        if labels.empty:
            errors.append("The dataset doens't have labels")
            return

        if labels.astype(str).str.strip().eq("").any():
            errors.append("Exist empty labels")

    def _validate_duplicates(self, df: pd.DataFrame, warnings: list[str]) -> None:
        if "text" not in df.columns:
            return

        duplicates = df["text"].duplicated().sum()

        if duplicates > 0:
            warnings.append(
                f"{duplicates} duplicate texts are found"
            )

    def _validate_cross_split_duplicates(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        errors: list[str],
    ) -> None:
        if not all(
            "text" in df.columns
            for df in (train, validation, test)
        ):
            return

        train_texts = set(train["text"].dropna().astype(str))
        validation_texts = set(
            validation["text"].dropna().astype(str)
        )
        test_texts = set(test["text"].dropna().astype(str))

        train_validation = train_texts & validation_texts
        train_test = train_texts & test_texts
        validation_test = validation_texts & test_texts

        if train_validation:
            errors.append(
                f"{len(train_validation)} texts appears in train and validation"
            )

        if train_test:
            errors.append(
                f"{len(train_test)} text appears in train and test"
            )

        if validation_test:
            errors.append(
                f"{len(validation_test)} text appears in validation and test"
            )
