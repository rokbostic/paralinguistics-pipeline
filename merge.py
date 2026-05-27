import csv
from pathlib import Path


TAGS = {
    "smeh":[
    "Belly laugh",
    "Chuckle, chortle",
    "Giggle",
    "Laughter",
    "Snicker",
    ],

    "jok": [
    "Crying, sobbing",
    "Wail, moan",
    ],

    "kaselj": [
    "Cough",
    "Throat clearing",
    ],

    "dihanje": [
    "Breathing",
    "Respiratory sounds",
    "Pant",
    "Gasp",
    "Sigh",
    "Wheeze",
    "Sniff",
    ]
}

REVERSE_TAGS = {
    value: key
    for key, values in TAGS.items()
    for value in values
}

def read_events_csv(path):

    if not path.exists():
        return None

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        events = list(reader)

    return events

def merge_with_overlap_markers(words, events):
    timeline = []

    EVENT_END = 0
    EVENT_START = 2
    WORD = 1

    # build timeline
    for e in events:

        tag = REVERSE_TAGS.get(e["event_label"])
        if tag is None:
            continue

        start = [e["offset"], EVENT_END, tag, None]
        end = [e["onset"], EVENT_START, tag, start]
        start[3] = end

        timeline.append(start)
        timeline.append(end)

    for w in words:
        word = [w["onset"], WORD, w["text"], None]
        timeline.append(word)

    timeline.sort(key=lambda x: (x[0], x[1], x[2]))

    for i in range(len(timeline)):
        timeline[i][0] = i

    output = []

    for i, typ, text, link in timeline:
        if typ == EVENT_START:
            if link[0] == i+1:
                output.append(f"[{text}]")
                continue

            output.append(f"<{text}>")
        
        if typ == EVENT_END:
            if link[0] == i-1:
                continue
            
            output.append(f"</{text}>")

        if typ == WORD:
            output.append(text)

    return " ".join(output)

def main():

    corpus_dir = Path("outputs/corpus")

    word_events_folder = Path("outputs/punctuate")
    sound_events_folder = Path("outputs/sed")
    
    output_file = Path("outputs/text")
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