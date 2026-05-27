'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''

EMOTION_MAP = {
    "neutral": "nevtralno",
    "happy": "sreča",
    "disgusted": "gnus",
    "sad": "žalost",
    # else: "drugo"
}

from funasr import AutoModel
from pathlib import Path

def main():
    # model="iic/emotion2vec_base"
    # model="iic/emotion2vec_base_finetuned"
    # model="iic/emotion2vec_plus_seed"
    # model="iic/emotion2vec_plus_base"
    model_id = "iic/emotion2vec_plus_large"

    model = AutoModel(
        model=model_id,
        hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    )

    corpus_dir = Path("outputs/corpus")

    output_file = Path("outputs/emotions")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in corpus_dir.rglob("*.flac")]
    rec_result = model.generate(files, granularity="utterance", extract_embedding=False)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in rec_result:
            item = r
            key = item["key"]
            emotion = max(zip(item["scores"], item["labels"]))[1].split("/")[-1]

            emotion = EMOTION_MAP.get(emotion, "drugo")
            f.write(f"{key} {emotion}\n")

if __name__ == "__main__":
    main()