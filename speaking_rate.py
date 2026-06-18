from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    audio_dir = Path("audio")
    aligner_dir = Path("outputs/aligner")

    filepaths = list(audio_dir.rglob("*.flac"))

    output_file = Path("outputs/speaking_rate")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for filepath in tqdm(filepaths):
            try:
                intervals = pd.read_csv(aligner_dir / filepath.with_suffix(".csv").name)
                onsets = intervals["onset"].tolist()
                offsets = intervals["offset"].tolist()

                duration = 0.
                for i in range(len(onsets)):
                    duration += offsets[i] - onsets[i]

                words_n = len(onsets)
                speaking_rate = words_n / duration

            except:
                speaking_rate = 2.
            f.write(f"{filepath.stem} {speaking_rate:.2f}\n")


if __name__ == "__main__":
    main()