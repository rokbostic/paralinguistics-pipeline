from pathlib import Path

from textgrid import TextGrid
import csv

aligner_dir = Path("outputs/aligner")

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
