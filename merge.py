import csv
from pathlib import Path

from tqdm import tqdm


MODEL_NAME = "ATST"


TAGS = {
    "smeh": [
        "Belly laugh",
        "Chuckle, chortle",
        "Giggle",
        "Laughter",
        "Snicker",
    ],
    "aplavz": [
        "Applause",
        "Clapping",
        "Cheering",
    ],
    "dihanje": [
        "Breathing",
        "Respiratory sounds",
        "Pant",
        "Gasp",
        "Sigh",
        "Wheeze",
        "Sniff",
    ],
    "medmet": [
        "medmet",
    ],
}

REVERSE_TAGS = {
    value: key
    for key, values in TAGS.items()
    for value in values
}

def merge_with_overlap_markers(words, events):
    timeline = []

    for e in events:
        tag = REVERSE_TAGS.get(e.get("event_label"))
        if tag is None: # we only allow the above listed tags
            continue

        timeline.append((e["onset"], 0, tag))

    for w in words:
        timeline.append((w["onset"], 1, w["text"]))

    timeline.sort(key=lambda x: (float(x[0]), x[1], x[2]))

    output = []

    used = set()

    for _, typ, text in timeline:

        if typ == 0:
            if text not in used:
                output.append(f"[{text}]")
                used.add(text)
        else:
            output.append(text)
            used = set()

    return " ".join(output)


def main():
    audio_dir = Path("audio")

    word_events_folder = Path("outputs/punctuate")
    sound_events_folder = Path("outputs/medmet_sed_"+MODEL_NAME)

    output_file = Path("outputs/text_"+MODEL_NAME)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for audio in tqdm(list(audio_dir.rglob("*.flac"))):
            words_file = word_events_folder / audio.with_suffix(".csv").name
            sounds_file = sound_events_folder / audio.with_suffix(".csv").name

            words = list(csv.DictReader(words_file.open(encoding="utf-8", newline=""))) if words_file.exists() else []
            sounds = list(csv.DictReader(sounds_file.open(encoding="utf-8", newline=""))) if sounds_file.exists() else []

            merged_text = merge_with_overlap_markers(words, sounds)

            f.write(f"{words_file.stem} {merged_text}\n")


if __name__ == "__main__":
    main()