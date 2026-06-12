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

from funasr import AutoModel
from pathlib import Path

def emotion2vec():
    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(model=model_id, hub="ms")

    corpus_dir = Path("outputs/corpus")

    output_file = Path("outputs/pipeline_emotions_emotion2vec")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in corpus_dir.rglob("*.flac")]
    rec_result = model.generate(files, granularity="utterance", extract_embedding=False)

    with open(output_file, "w", encoding="utf-8") as f:
        for r in rec_result:
            item = r
            key = item["key"]
            emotion = max(zip(item["scores"], item["labels"]))[1].split("/")[-1]

            emotion = EMOTION_MAP.get(emotion, "drugo")
            f.write(f"{key} {emotion}\n")

import re

def sensevoice_emotions():
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        hub="ms",
        ban_emo_unk=True,
    )

    corpus_dir = Path("outputs/corpus")
    output_file = Path("outputs/pipeline_emotions_sensevoice")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(corpus_dir.rglob("*.flac"))

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

import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

def emotions(model_str):

    if model_str == "hubert":
        model_name = "superb/hubert-large-superb-er"

    elif model_str == "wav2vec2":
        model_name = "superb/wav2vec2-large-superb-er"

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()
    print(model.config.id2label)

    corpus_dir = Path("outputs/corpus")

    output_file = Path("outputs/pipeline_emotions_"+model_str) 
    output_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(corpus_dir.rglob("*.flac"))

    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in files:
            # Load 24 kHz FLAC and resample to 16 kHz
            audio, _ = librosa.load(
                file_path,
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

            key = file_path.stem
            f.write(f"{key} {emotion}\n")

if __name__ == "__main__":

    # IMPLEMENTED emotion2vec, sensevoice, hubert, wav2vec2 

    MODEL_NAME = "hubert"

    if MODEL_NAME == "emotion2vec":
        emotion2vec()
    elif MODEL_NAME == "sensevoice":
        sensevoice_emotions()
    elif MODEL_NAME == "wav2vec2":
        emotions("wav2vec2")
    elif MODEL_NAME == "hubert":
        emotions("hubert")
