import csv
from pathlib import Path


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


def read_events_csv(path):
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def merge_with_overlap_markers(words, events):
    timeline = []

    EVENT_END = 0
    WORD = 1
    EVENT_START = 2

    for e in events:
        tag = REVERSE_TAGS.get(e.get("event_label"))
        if tag is None:
            continue

        start = [e["onset"], EVENT_START, tag, None]
        end = [e["offset"], EVENT_END, tag, start]
        start[3] = end

        timeline.append(start)
        timeline.append(end)

    for w in words:
        timeline.append([w["onset"], WORD, w["text"], None])

    timeline.sort(key=lambda x: (float(x[0]), x[1], x[2]))

    for i in range(len(timeline)):
        timeline[i][0] = i

    output = []

    for i, typ, text, link in timeline:
        if typ == EVENT_START:
            if link[0] == i + 1:
                output.append(f"[{text}]")
            else:
                output.append(f"<{text}>")

        elif typ == EVENT_END:
            if link[0] == i - 1:
                continue
            output.append(f"</{text}>")

        elif typ == WORD:
            output.append(text)

    return " ".join(output)


def main():
    corpus_dir = Path("outputs/corpus")

    word_events_folder = Path("outputs/punctuate")
    sound_events_folder = Path("outputs/medmet_sed")

    output_file = Path("outputs/pipeline_text")
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