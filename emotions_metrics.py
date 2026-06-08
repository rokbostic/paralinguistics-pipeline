import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

def calculate_metrics(pred_dir):
    target = pd.read_csv(
        Path("annotated_target_emotions"),
        sep=r"\s+",
        header=None,
        names=["ID", "true"]
    )

    pred = pd.read_csv(
        pred_dir,
        sep=r"\s+",
        header=None,
        names=["ID", "pred"]
    )

    df = target.merge(pred, on="ID")

    y_true = df["true"]
    y_pred = df["pred"]

    classes = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0
    )

    print(f"\n--- Evaluation Results for: {pred_dir.name} ---")
    for i, label in enumerate(classes):
        print(f"Class: {label}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-score:  {f1[i]:.4f}")
        print(f"  Support:   {support[i]}")
        print()

if __name__ == "__main__":

    #pred_dir = Path("outputs/gemini_emotions")
    pred_dir = Path("outputs/pipeline_emotions")
    calculate_metrics(pred_dir)