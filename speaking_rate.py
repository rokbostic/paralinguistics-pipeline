from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    audio_dir = Path("audio")
    aligner_dir = Path("outputs/aligner")

    filepaths = list(audio_dir.rglob("*.flac"))

    output_file = Path("outputs/utt2speakingrate")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for filepath in tqdm(filepaths):
            try:
                intervals = pd.read_csv(aligner_dir / f"{filepath.stem}.csv")

                duration = (intervals["offset"] - intervals["onset"]).sum()
                speaking_rate = len(intervals) / duration

            except (FileNotFoundError, KeyError, pd.errors.EmptyDataError, ZeroDivisionError):
                speaking_rate = 2.0

            f.write(f"{filepath.stem} {speaking_rate:.2f}\n")


if __name__ == "__main__":
    main()