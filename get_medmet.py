BATCH_SIZE = 4 * 32

from pathlib import Path
from tqdm import tqdm

import nemo.collections.asr as nemo_asr


def main():
    
    audio_dir = Path("audio")

    output_file = Path("outputs/medmet")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    done = {line.partition(" ")[0] for line in output_file.open()} if output_file.exists() else set()

    filepaths = sorted(audio_dir.rglob("*.flac"))
    filepaths = [p for p in filepaths if p.stem not in done]

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path="resources/nemo-train-asr-char.nemo"
    )

    with output_file.open("a") as f:
        with tqdm(total=len(filepaths), desc="Transcribing", unit="file") as pbar:
            for i in range(0, len(filepaths), BATCH_SIZE):
                batch = filepaths[i:i + BATCH_SIZE]

                results = model.transcribe([str(p) for p in batch])

                for audio, transcription in zip(batch, results):
                    text = getattr(transcription, "text", transcription)
                    f.write(f"{audio.stem} {text}\n")

                f.flush()
                pbar.update(len(batch))

if __name__ == "__main__":
    main()