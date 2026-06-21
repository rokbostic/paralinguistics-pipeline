BATCH_SIZE = 5000

import subprocess
import shutil
from pathlib import Path
import tempfile
from tqdm import tqdm

def main(texts_file, output_dir):
    audio_dir = Path("audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(audio_dir.rglob("*.flac"))

    done_stems = set()
    for filepath in filepaths:
        target_file = output_dir / filepath.with_suffix(".TextGrid").name
        if target_file.exists():
            done_stems.add(filepath.stem)
    
    filepaths = [p for p in filepaths if p.stem not in done_stems]

    dictionary = Path("resources/dictionary.txt")
    acoustic_model = Path("resources/acoustic_model.zip")

    with open(texts_file) as f:
        texts = {utt: txt for utt, txt in ([line.split()[0], " ".join(line.split()[1:])] for line in f)}

    def align(batch_files):
        with tempfile.TemporaryDirectory() as temp_folder:
            temp_folder = Path(temp_folder)

            for pathfile in batch_files:
                shutil.copy2(pathfile, temp_folder / pathfile.name)

                text = texts[pathfile.stem]

                txt_path = (temp_folder / pathfile.name).with_suffix(".txt")
                txt_path.write_text(text)

            com = [
                "mfa", "align",
                str(temp_folder),
                str(dictionary),
                str(acoustic_model),
                str(output_dir),
                "--clean",
                "--overwrite",
                "--single_speaker",
                "--num_jobs", "4",
            ]

            subprocess.run(com, check=True)

    with tqdm(total=len(filepaths), desc=f"Aligning: ", unit="file") as pbar:
        for i in range(0, len(filepaths), BATCH_SIZE):
            batch = filepaths[i:i + BATCH_SIZE]
            align(batch)
            #shutil.rmtree("~/Documents/MFA")
            pbar.update(len(batch))


if __name__ == "__main__":
    main(Path("text"), Path("outputs/aligner"))
    main(Path("outputs/medmet"), Path("outputs/medmet_aligner"))