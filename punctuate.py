from pathlib import Path
import csv
import pandas as pd

import re
from textgrid import TextGrid
import csv

from tqdm import tqdm

def textgrid2csv(aligner_dir: Path):

    filepaths = sorted(aligner_dir.rglob("*.TextGrid"))

    done_stems = set()
    for filepath in filepaths:
        target_file = filepath.with_suffix(".csv")
        if target_file.exists():
            done_stems.add(filepath.stem)
    
    filepaths = [p for p in filepaths if p.stem not in done_stems]

    for filepath in tqdm(filepaths):
        (
            pd.DataFrame(
                ((i.mark, i.minTime, i.maxTime)
                 for i in TextGrid.fromFile(filepath)[0]
                 if i.mark.strip()),
                columns=["text", "onset", "offset"],
            )
            .to_csv(filepath.with_suffix(".csv"), index=False)
        )

def main():

    aligner_dir = Path("outputs/aligner")
    #textgrid2csv(aligner_dir)

    aligner_dir = Path("outputs/medmet_aligner")
    #textgrid2csv(aligner_dir)

    with open("text") as f:
        text = {utt: txt for utt, txt in ([line.split()[0], " ".join(line.split()[1:])] for line in f)}

    audio_dir = Path("audio")

    word_events_folder = Path("outputs/aligner")

    output_folder = Path("outputs/punctuate")
    output_folder.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(audio_dir.rglob("*.flac"))

    done_stems = set()
    for filepath in filepaths:
        target_file = output_folder / filepath.with_suffix(".csv").name
        if target_file.exists():
            done_stems.add(filepath.stem)
    
    filepaths = [p for p in filepaths if p.stem not in done_stems]

    for filepath in tqdm(filepaths):

        try:

            words_file = word_events_folder / filepath.with_suffix(".csv").name
            output_file = output_folder / filepath.with_suffix(".csv").name
            utt = words_file.stem

            punctuated = re.split(r"\s+",text[utt])

            new_punctuated = []

            preamble = ""
            for word in punctuated:
                if re.fullmatch(r"[^\w€']+", word): # ONLY PUNCTUATION
                    if len(new_punctuated) > 0:
                        new_punctuated[-1] += f" {word}"
                    else:
                        preamble += f"{word} "
                else:
                    parts = re.findall(r"[\w€']+(?:[^\w€']+)?", word)
                    if preamble:
                        parts[0] = preamble + parts[0]
                        preamble = ""
                    for part in parts:
                        new_punctuated.append(part)
                    
            punctuated = new_punctuated

            if words_file.exists():
                df = pd.read_csv(words_file, dtype=str, keep_default_na=False)
            else:
                df = pd.DataFrame({
                    "text": punctuated,
                    "onset": [0 for _ in range(len(punctuated))],
                    "offset": [0 for _ in range(len(punctuated))],
                })
                df.to_csv(output_file, index=False)
                continue
            unpunctuated = df["text"].to_list()

            new_punctuated = []
            cringe_sum = 0
            for i, word in enumerate(unpunctuated):
                new_punctuated.append(punctuated[cringe_sum + i])
                cringe = len([p for p in word.split("-") if p]) - 1
                for o in range(cringe):
                    new_punctuated[-1] += punctuated[cringe_sum + i + o+1]
                cringe_sum += cringe
            
            punctuated = new_punctuated
            
            df["text"] = punctuated
            df.to_csv(output_file, index=False)

        except Exception as e:
            print(e)
            print(f"{len(unpunctuated)}, {len(punctuated)}")
            print(text[utt])
            print(punctuated)
            print(unpunctuated)
            return




if __name__ == "__main__":
    main()