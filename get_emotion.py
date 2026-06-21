from funasr import AutoModel
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from tqdm import tqdm

from pathlib import Path
import re

EMOTION_MAP = {
    "neutral": "nevtralno",
    "Neutral": "nevtralno",
    "NEUTRAL": "nevtralno",
    "EMO_NEUTRAL": "nevtralno",
    "neu": "nevtralno",

    "happy": "sreča",
    "Happiness": "sreča",
    "HAPPY": "sreča",
    "EMO_HAPPY": "sreča",
    "hap": "sreča",
    
    "angry": "jeza",
    "Anger": "jeza",
    "ANGRY": "jeza",
    "EMO_ANGRY": "jeza",
    "ang": "jeza",
    
    "sad": "žalost",
    "Sadness": "žalost",
    "SAD": "žalost",
    "EMO_SAD": "žalost",

    # else: "drugo"
}

BATCH_SIZE = 64

def emotions_emotion2vec():
    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(model=model_id, hub="ms")

    audio_dir = Path("audio")

    output_file = Path("outputs/emotions_emotion2vec")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    done = {line.partition(" ")[0] for line in output_file.open()} if output_file.exists() else set()

    filepaths = sorted(str(f) for f in audio_dir.rglob("*.flac"))
    filepaths = [p for p in filepaths if Path(p).stem not in done]

    with open(output_file, "a", encoding="utf-8") as f:
        with tqdm(total=len(filepaths), desc="Getting emotions", unit="file") as pbar:

            for i in range(0, len(filepaths), BATCH_SIZE):
                batch = filepaths[i:i + BATCH_SIZE]
                result = model.generate(batch, granularity="utterance", extract_embedding=False)

                for item in result:
                    key = item["key"]
                    emotion = max(zip(item["scores"], item["labels"]))[1].split("/")[-1]
                    emotion = EMOTION_MAP.get(emotion, "drugo")
                    f.write(f"{key} {emotion}\n")

                f.flush()
                pbar.update(len(batch))






def emotions_sensevoice():
    model_id = "iic/SenseVoiceSmall"
    model = AutoModel(model=model_id, hub="ms", ban_emo_unk=True,)

    audio_dir = Path("audio")

    output_file = Path("outputs/emotions_sensevoice")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(str(f) for f in audio_dir.rglob("*.flac"))

    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in files:
            result = model.generate(input=str(file_path))

            text = result[0].get("text", "")

            match = re.search(
                r"<\|(HAPPY|SAD|ANGRY|NEUTRAL|UNKNOWN|EMO_UNKNOWN)\|>",
                text
            )

            raw_emotion = match.group(1) if match else "UNKNOWN"
            emotion = EMOTION_MAP.get(raw_emotion, "drugo")

            key = result[0].get("key", file_path.stem)

            f.write(f"{key} {emotion}\n")

def emotions_huggingface(model_str):

    if model_str == "hubert":
        model_name = "superb/hubert-large-superb-er"

    elif model_str == "wav2vec2":
        model_name = "superb/wav2vec2-large-superb-er"

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()
    print(model.config.id2label)

    audio_dir = Path("audio")

    output_file = Path("outputs/emotions_"+model_str) 
    output_file.parent.mkdir(parents=True, exist_ok=True)

    done = {line.partition(" ")[0] for line in output_file.open()} if output_file.exists() else set()

    filepaths = sorted(str(f) for f in audio_dir.rglob("*.flac"))
    filepaths = [p for p in filepaths if Path(p).stem not in done]

    with open(output_file, "a", encoding="utf-8") as f:
        for filepath in tqdm(filepaths):
            # Load 24 kHz FLAC and resample to 16 kHz
            audio, _ = librosa.load(
                filepath,
                sr=16000,
                mono=True
            )

            inputs = extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            with torch.no_grad():
                logits = model(**inputs).logits

            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()

            emotion = model.config.id2label[pred_id].lower()
            emotion = emotion.split("/")[-1]

            emotion = EMOTION_MAP.get(emotion, "drugo")

            key = Path(filepath).stem
            f.write(f"{key} {emotion}\n")
            f.flush()

if __name__ == "__main__":

    # IMPLEMENTED emotion2vec, sensevoice, hubert, wav2vec2 
    MODEL_NAME = "emotion2vec"

    if MODEL_NAME == "emotion2vec": # implements batching
        emotions_emotion2vec()


    elif MODEL_NAME == "sensevoice":
        emotions_sensevoice()
    elif MODEL_NAME == "wav2vec2":
        emotions_huggingface("wav2vec2")
    elif MODEL_NAME == "hubert":
        emotions_huggingface("hubert")
