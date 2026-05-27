import subprocess
import shutil
from pathlib import Path
import tempfile

BATCH_SIZE = 5000

def main():
    corpus_dir = Path("outputs/corpus")

    output_dir = Path("outputs/aligner")
    output_dir.mkdir(parents=True, exist_ok=True)

    dictionary = Path("resources/dictionary.txt")
    acoustic_model = Path("resources/acoustic_model.zip")

    files = []

    def align(batch_files):
        if not batch_files:
            return

        with tempfile.TemporaryDirectory() as temp_folder:
            temp_folder = Path(temp_folder)

            # Copy files into temp folder
            for src in batch_files:
                if src.exists():
                    shutil.copy2(src, temp_folder / src.name)

            com = [
                "mfa", "align",
                str(temp_folder),
                str(dictionary),
                str(acoustic_model),
                str(output_dir),
                "--clean",
                "--overwrite",
                "--single_speaker",
                "--num_jobs", "2",
            ]

            print(f"Running: {' '.join(com)}")
            subprocess.run(com, check=True)
            print("Alignment complete for batch!")

    for filepath in corpus_dir.rglob("*.flac"):
        output_csv = output_dir / filepath.with_suffix(".TextGrid").name

        if output_csv.exists():
            continue

        files.append(filepath)
        files.append(filepath.with_suffix(".txt"))

        if len(files) >= BATCH_SIZE:
            align(files)
            files.clear()

    align(files)

if __name__ == "__main__":
    main()