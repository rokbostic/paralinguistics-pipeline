"""
Creates a corpus in outputs/aligner which the MFA can read in the next step of the pipeline.

The files in audio are expected to be named the same as their utterance name in wav.scp with a .flac suffix.
"""

from pathlib import Path

def create_corpus(input_fil: Path, output_dir: Path):

    with open(input_fil) as f:
        texts = {utt: txt for utt, txt in ([line.split()[0], " ".join(line.split()[1:])] for line in f)}

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = Path("audio")
    for filepath in audio_dir.rglob("*.flac"):
        utt = filepath.with_suffix("").name
        text = texts[utt]

        sym_path = output_dir / filepath.name
        txt_path = output_dir / filepath.with_suffix(".txt").name

        if sym_path.exists():
            continue

        sym_path.symlink_to(filepath.resolve())
        txt_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    create_corpus(Path("text"), Path("outputs/corpus"))