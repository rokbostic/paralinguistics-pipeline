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

    # Micro average
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    # Macro average
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("Micro Average")
    print(f"  Precision: {p_micro:.4f}")
    print(f"  Recall:    {r_micro:.4f}")
    print(f"  F1-score:  {f1_micro:.4f}")
    print()

    print("Macro Average")
    print(f"  Precision: {p_macro:.4f}")
    print(f"  Recall:    {r_macro:.4f}")
    print(f"  F1-score:  {f1_macro:.4f}")
    print()


if __name__ == "__main__":
    

    paths = [
        Path("outputs/gemini_emotions"),
        Path("outputs/pipeline_emotions_sensevoice"),
        Path("outputs/pipeline_emotions_wav2vec2"),
        Path("outputs/pipeline_emotions_emotion2vec"),
        Path("outputs/pipeline_emotions_hubert")
        ]

    for path in paths:
    
        data = calculate_metrics(path)
        print(data)