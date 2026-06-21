from pathlib import Path

def kaldi_to_dict(filepath, is_float=True):
    return {
        k: float(v) if is_float else v
        for k, v in (
            line.split(maxsplit=1)
            for line in filepath.read_text(encoding="utf-8").splitlines()
            if line
        )
    }

def mean(values):
    return sum(values) / len(values)

def main():

    nevtralno_count = 0
    sreca_count = 0
    jeza_count = 0
    zalost_count = 0
    undecided_count = 0
    full_count = 0

    emotions_list = [
        kaldi_to_dict(f, False)
        for f in sorted(Path("outputs").glob("*emotions*"))
    ]

    speaking_rates = kaldi_to_dict(Path("outputs/utt2speakingrate"))
    speaking_rate_avg = mean(speaking_rates.values())

    arousal_rates = kaldi_to_dict(Path("utt2arousal"))
    arousal_avg = mean(arousal_rates.values())

    with open("outputs/instructions", "w", encoding="utf-8") as f:
        for utt in emotions_list[0]:
            emotions = {emotion[utt] for emotion in emotions_list}
            decided_emotion = next(iter(emotions)) if len(emotions) == 1 else None



            speed = speaking_rates.get(utt, speaking_rate_avg)
            arousal = arousal_rates.get(utt, arousal_avg)

            speed_str = "hitro" if speed > speaking_rate_avg else "počasi"
            arousal_str = "vzburjeno" if arousal > arousal_avg else "nevzburjeno"

            emotion_text = f" in v čustvu {decided_emotion}" if decided_emotion else ""
            instruction = f"Govori {speed_str} in {arousal_str}{emotion_text}."

            f.write(f"{utt} {instruction}\n")

            
            if decided_emotion is None:
                undecided_count += 1
            elif decided_emotion == "nevtralno":
                nevtralno_count += 1
            elif decided_emotion == "sreča":
                sreca_count += 1
            elif decided_emotion == "jeza":
                jeza_count += 1
            elif decided_emotion == "žalost":
                zalost_count += 1
            full_count += 1
    
    print(f"Zastopanost razredov je:\nnevtralno: {nevtralno_count/full_count:.2f}%\njeza: {jeza_count/full_count:.2f}%\nsreca: {sreca_count/full_count:.2f}%\nzalost: {zalost_count/full_count:.2f}\nnedoloceno (modeli se ne strinjajo): {undecided_count/full_count:.2f}%")


if __name__ == "__main__":
    main()