import nemo.collections.asr as nemo_asr # pip install nemo_toolkit[asr]

from create_corpus import create_corpus

from pathlib import Path

def main():
    
    audio_dir = Path("audio")

    output_file = Path("outputs/medmet")

    if not output_file.exists():
        model = nemo_asr.models.ASRModel.restore_from(restore_path="resources/nemo-train-asr-char.nemo")

        filepaths = [str(p) for p in audio_dir.glob("*.flac")]
        results = model.transcribe(filepaths)

        with open(output_file, "w") as f:
            for audio, transcription in zip(filepaths, results):
                f.write(f"{Path(audio).stem} {transcription.text}\n")
        
    create_corpus(output_file, Path("outputs/medmet_corpus"))

if __name__ == "__main__":
    main()