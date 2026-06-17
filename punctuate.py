from pathlib import Path
import csv
import pandas as pd

import re
from textgrid import TextGrid
import csv

def textgrid2csv(aligner_dir: Path):
    for file in aligner_dir.glob("*.TextGrid"):
        
        tg = TextGrid.fromFile(file)
        tier = tg[0] # words - 0, phonemes - 1

        output_file = file.with_suffix(".csv")

        # Write CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "onset", "offset"])

            for interval in tier:
                if interval.mark.strip():  # skip empty labels
                    writer.writerow([
                        interval.mark,
                        interval.minTime,
                        interval.maxTime
                    ])

def get_punctuated_orig(orig):
    word = ""
    words = []

    prev_alpha = True
    for c in orig:
        if c.isalpha():
            if prev_alpha == False:
                words.append(word)
                word = ""
            prev_alpha = True
        else:
            prev_alpha = False

        word += c
    words.append(word)

    good = False
    for c in words[0]:
        if c.isalpha():
            good = True
            break
    if not good:
        words.pop(0)

    return words

def main():

    aligner_dir = Path("outputs/aligner")
    textgrid2csv(aligner_dir)

    aligner_dir = Path("outputs/medmet_aligner")
    textgrid2csv(aligner_dir)

    with open("text") as f:
        text = {utt: txt for utt, txt in ([line.split()[0], " ".join(line.split()[1:])] for line in f)}

    corpus_dir = Path("outputs/corpus")

    word_events_folder = Path("outputs/aligner")

    output_folder = Path("outputs/punctuate")
    output_folder.mkdir(parents=True, exist_ok=True)

    for audio in corpus_dir.glob("*.flac"):

        words_file = word_events_folder / audio.with_suffix(".csv").name
        output_file = output_folder / audio.with_suffix(".csv").name

        utt = words_file.stem

        if words_file.exists():
            aligned = pd.read_csv(words_file)

            punctuated_original = get_punctuated_orig(text[utt])
            aligned["text"] = punctuated_original
        else:
            words = text[utt].split("\n")
            aligned = {"text": words, "onset": [0 for _ in range(len(words))], "offset": [0 for _ in range(len(words))]}           
            aligned = pd.DataFrame(aligned)


        aligned.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()