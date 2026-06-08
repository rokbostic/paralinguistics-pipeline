from pathlib import Path

import pandas as pd


def main():
    medmet_dir = Path("outputs/medmet_aligner")
    sed_dir = Path("outputs/sed")

    output_dir = Path("outputs/medmet_sed")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(medmet_dir.glob("*.csv"))

    for file in files:
        if file.stem == "alignment_analysis":
            continue
        utt = file.stem

        medmet_events = pd.read_csv(file)

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