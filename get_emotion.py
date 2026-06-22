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

BATCH_SIZE = 128

def emotions_emotion2vec():
    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(
        model=model_id,
        hub="ms",
        device="cuda:0",
    )
    audio_dir = Path("audio")

    output_file = Path("outputs/emotions_emotion2vec_prob")
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
                    prob, best_label = max(
                        zip(item["scores"], item["labels"]),
                        key=lambda x: x[0]
                    )

                    emotion = best_label.split("/")[-1]
                    emotion = EMOTION_MAP.get(emotion, "drugo")

                    f.write(f"{key} {emotion} {prob:.6f}\n")

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





def emotions_huggingface(model_str, batch_size=16):

    if model_str == "hubert":
        model_name = "superb/hubert-large-superb-er"
    elif model_str == "wav2vec2":
        model_name = "superb/wav2vec2-large-superb-er"
    else:
        raise ValueError(f"Unsupported model: {model_str}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    extractor = AutoFeatureExtractor.from_pretrained(model_name)

    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print(model.config.id2label)

    audio_dir = Path("audio")

    output_file = Path(f"outputs/emotions_{model_str}")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    done = (
        {line.partition(" ")[0] for line in output_file.open()}
        if output_file.exists()
        else set()
    )

    filepaths = sorted(audio_dir.rglob("*.flac"))
    filepaths = [p for p in filepaths if p.stem not in done]

    print(f"Remaining files: {len(filepaths)}")

    with open(output_file, "a", encoding="utf-8") as f:

        batch_audio = []
        batch_keys = []

        for filepath in tqdm(filepaths):

            audio, _ = librosa.load(
                filepath,
                sr=16000,
                mono=True
            )

            batch_audio.append(audio)
            batch_keys.append(filepath.stem)

            if len(batch_audio) < batch_size:
                continue

            inputs = extractor(
                batch_audio,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                logits = model(**inputs).logits

            pred_ids = logits.argmax(dim=-1).tolist()

            for key, pred_id in zip(batch_keys, pred_ids):

                emotion = model.config.id2label[pred_id].lower()
                emotion = emotion.split("/")[-1]
                emotion = EMOTION_MAP.get(emotion, "drugo")

                f.write(f"{key} {emotion}\n")

            f.flush()

            batch_audio.clear()
            batch_keys.clear()

        # Process final partial batch
        if batch_audio:

            inputs = extractor(
                batch_audio,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                logits = model(**inputs).logits

            pred_ids = logits.argmax(dim=-1).tolist()

            for key, pred_id in zip(batch_keys, pred_ids):

                emotion = model.config.id2label[pred_id].lower()
                emotion = emotion.split("/")[-1]
                emotion = EMOTION_MAP.get(emotion, "drugo")

                f.write(f"{key} {emotion}\n")

            f.flush()




if __name__ == "__main__":

    # IMPLEMENTED emotion2vec, sensevoice, hubert, wav2vec2 
    MODEL_NAME = "emotion2vec"

    if MODEL_NAME == "emotion2vec": # implements batching + probs
        emotions_emotion2vec()


    elif MODEL_NAME == "sensevoice":
        emotions_sensevoice()
    elif MODEL_NAME == "wav2vec2":
        emotions_huggingface("wav2vec2")
    elif MODEL_NAME == "hubert":
        emotions_huggingface("hubert")
