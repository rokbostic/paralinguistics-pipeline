from pathlib import Path
import json
from eval.eval import (
    evaluate,
    EvaluationConfig,
    format_output_by_tag_type,
    tags_1226_21,
)

def calculate_metrics(pred_dir):
    target_dir = Path("annotated_target_text")

    pred_texts = dict(line.split(maxsplit=1) for line in pred_dir.read_text().splitlines())
    target_texts = dict(line.split(maxsplit=1) for line in target_dir.read_text().splitlines())

    pred_data = []
    target_data = []

    for utt in pred_texts:
        pred_text = pred_texts[utt]
        target_text = target_texts[utt]
        
        sample = utt

        pred_data.append({
            "audio": {"path": sample},
            "sentence": pred_text
        })

        target_data.append({
            "audio": {"path": sample},
            "sentence": target_text
        })

    pred_jsonl = Path("predicted.jsonl")
    target_jsonl = Path("target.jsonl")


    with pred_jsonl.open("w", encoding="utf-8") as f:
        for item in pred_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with target_jsonl.open("w", encoding="utf-8") as f:
        for item in target_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


    config = EvaluationConfig(
        include_tags=tags_1226_21,
        big=False
    )

    results = evaluate(
        true_path=[str(target_jsonl)],
        pred_path=str(pred_jsonl),
        config=config,
        eval_type="sequence",
        tag_type="by_type"
    )

    print(format_output_by_tag_type(results))



if __name__ == "__main__":

    #pred_dir = Path("outputs/gemini_text")
    pred_dir = Path("outputs/pipeline_text")
    calculate_metrics(pred_dir)