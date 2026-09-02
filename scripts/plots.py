from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

sns.set_theme(style="whitegrid")

def plot_yb_confusion_matrix(
    model,
    X:np.ndarray,
    y_true:np.ndarray,
    class_names:list[str],
    split:str = "test",
    model_name: str = "",
    save_path:str | Path | None = None,
    show:bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(7,6))

    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"

    cm = ConfusionMatrix(model, classes = class_names, is_fitted=True, cmap="Blues",ax=ax)
    cm.score(X, y_true)
    
    name_suffix = f"_{model_name}" if model_name else ""
    path = _prepare_save_path(save_path, f"confusion_matrix_yb_{split}{name_suffix}.png")
    cm.show(outpath=str(path),clear_figure=False)
    return path

def plot_yb_classification_report(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    split: str = "test",
    model_name: str = "",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    
    visualizer = ClassificationReport(
        model, 
        classes=class_names, 
        support=True, 
        is_fitted=True,
        cmap="YlGnBu",
        ax=ax
    )
    
    visualizer.score(X, y_true)
    
    name_suffix = f"_{model_name}" if model_name else ""
    path = _prepare_save_path(save_path, f"classification_report_yb_{split}{name_suffix}.png")
    visualizer.show(outpath=str(path), clear_figure=False)
    if show:
        plt.show()
    plt.close(fig)
    return path


def _prepare_save_path(save_path: str | Path | None, default_name: str) -> Path:
    if save_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_DIR / default_name
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    split: str = "test",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    """
    Gera um heatmap da matriz de confusão em
    termos de contagem absoluta
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"Matriz de confusão — {split}")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    path = _prepare_save_path(save_path, f"confusion_matrix_{split}.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    split: str = "test",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    """
    Gera um heatmap com precision/recall/f1-score por classe.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows = class_names + ["macro avg", "weighted avg"]
    metrics = ["precision", "recall", "f1-score"]
    data = np.array([[report[row][metric] for metric in metrics] for row in rows])

    fig, ax = plt.subplots(figsize=(6, 0.55 * len(rows) + 1.5))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=metrics,
        yticklabels=rows,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Relatório de classificação—{split} (accuracy={report['accuracy']:.2f})")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    fig.tight_layout()

    path = _prepare_save_path(save_path, f"classification_report_{split}.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path:
    """
    Gera um gráfico de barras comparando accuracy e f1_macro
    entre candidatos. É usado pelo train.py para comparar
    logreg vs svm na validação.
    """
    names = list(results.keys())
    accuracy = [results[n]["accuracy"] for n in names]
    f1_macro = [results[n]["f1_macro"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, accuracy, width, label="accuracy", color="#4C72B0")
    ax.bar(x + width / 2, f1_macro, width, label="f1_macro", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Comparação de modelos na validação")
    ax.legend()

    for i, (a, f) in enumerate(zip(accuracy, f1_macro)):
        ax.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=9)

    fig.tight_layout()

    path = _prepare_save_path(save_path, "model_comparison.png")
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path
