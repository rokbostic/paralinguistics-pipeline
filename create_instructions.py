from pathlib import Path

def mean(values):
    return sum(values) / len(values)

def main():
    emotion_dict = {}

    with open(Path("outputs/emotion2vec_prob"), "r", encoding="utf-8") as f:
        for line in f:
            file_id, label, score = line.strip().split()
            emotion_dict[file_id] = {
                "label": label,
                "score": float(score)
            }

    speaking_rates = {
        k: float(v)
        for k, v in (
            line.split(maxsplit=1)
            for line in Path("outputs/utt2speakingrate").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    speaking_rate_avg = mean(speaking_rates.values())

    arousal_rates = {
        k: float(v)
        for k, v in (
            line.split(maxsplit=1)
            for line in Path("utt2arousal").read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    arousal_avg = mean(arousal_rates.values())

    with open("outputs/instructions", "w", encoding="utf-8") as f:
        for utt in emotion_dict:
            emotion = emotion_dict[utt]["label"]
            prob = emotion_dict[utt]["score"]
            speed = speaking_rates.get(utt, speaking_rate_avg)
            arousal = arousal_rates.get(utt, arousal_avg)

            speed_str = "hitro" if speed > speaking_rate_avg else "počasi"
            arousal_str = "vzburjeno" if arousal > arousal_avg else "nevzburjeno"

            emotion_text = f" in v čustvu {emotion}" if prob > .9 else ""
            instruction = f"Govori {speed_str} in {arousal_str}{emotion_text}."

            f.write(f"{utt} {instruction}\n")

if __name__ == "__main__":
    main()