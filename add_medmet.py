from pathlib import Path

import pandas as pd
from tqdm import tqdm 

MODEL_NAME = "ATST"

def main():
    medmet_dir = Path("outputs/medmet_aligner")

    sed_dir = Path("outputs/sed_"+MODEL_NAME)

    output_dir = Path("outputs/medmet_sed_"+MODEL_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(medmet_dir.glob("*.csv"))

    done_stems = set()
    for filepath in filepaths:
        target_file = output_dir / filepath.with_suffix(".csv").name
        if target_file.exists():
            done_stems.add(filepath.stem)
    
    filepaths = [p for p in filepaths if p.stem not in done_stems]


    for filepath in tqdm(filepaths):
        if filepath.stem == "alignment_analysis":
            continue
        utt = filepath.stem

        medmet_events = pd.read_csv(filepath)

        sed_file = sed_dir / f"{utt}.csv"
        output_file = output_dir / f"{utt}.csv"

        output_events = pd.read_csv(sed_file)

        medmet_mask = medmet_events["text"].isin(["e", "ee", "eee"])

        additions = medmet_events.loc[medmet_mask, ["onset", "offset"]].copy()
        additions["event_label"] = "medmet"
        additions["filename"] = "o.wav"

        output_events = pd.concat(
            [output_events, additions[["event_label", "onset", "offset", "filename"]]],
            ignore_index=True,
        )

        output_events.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()