from pathlib import Path
import random


audio_dir = Path("audio")

filepaths = list(audio_dir.rglob("*.flac"))
random.shuffle(filepaths)

filepath = filepaths[0]
utt = filepath.stem



target_dir = Path("annotated_target_text")
target_texts = dict(line.split(maxsplit=1) for line in target_dir.read_text().splitlines())

gemini_dir = Path("outputs/gemini_text")
gemini_texts = dict(line.split(maxsplit=1) for line in gemini_dir.read_text().splitlines())

pipeline_dir = Path("outputs/pipeline_text")
pipeline_texts = dict(line.split(maxsplit=1) for line in pipeline_dir.read_text().splitlines())

print(utt)
print()
print(target_texts[utt])
print()
print(gemini_texts[utt])
print()
print(pipeline_texts[utt])