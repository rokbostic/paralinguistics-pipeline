import csv
from pathlib import Path


MODEL_NAME = "BEATS"


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

DISCRETE = {"dihanje", "medmet"}

REVERSE_TAGS = {
    value: key
    for key, values in TAGS.items()
    for value in values
}


def read_events_csv(path):
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_with_overlap_markers(words, events):
    timeline = []

    for e in events:
        tag = REVERSE_TAGS.get(e.get("event_label"))
        if tag is None:
            continue

        timeline.append((e["onset"], 0, tag))

    for w in words:
        timeline.append((w["onset"], 1, w["text"]))

    timeline.sort(key=lambda x: (float(x[0]), x[1], x[2]))

    output = []

    used = set()

    for i, typ, text in timeline:

        if typ == 0:
            if text not in used:
                output.append(f"[{text}]")
                used.add(text)
        else:
            output.append(text)
            used = set()

    return " ".join(output)


def main():
    corpus_dir = Path("outputs/corpus")

    word_events_folder = Path("outputs/punctuate")
    sound_events_folder = Path("outputs/medmet_sed_"+MODEL_NAME)

    output_file = Path("outputs/pipeline_text_"+MODEL_NAME)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for audio in corpus_dir.glob("*.flac"):
            words_file = word_events_folder / audio.with_suffix(".csv").name
            sounds_file = sound_events_folder / audio.with_suffix(".csv").name

            words = read_events_csv(words_file)
            sounds = read_events_csv(sounds_file)

            merged_text = merge_with_overlap_markers(words, sounds)

            f.write(f"{words_file.stem} {merged_text}\n")


if __name__ == "__main__":
    main()