import argparse
import torch
import torchaudio

from helpers import audioset_classes
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import csv

from config import SED_MODEL_NAME, SED_THRESHOLD, SED_MEDIAN_FILTER, SED_REVERSE_TAGS


SAMPLE_RATE = 16000
SEGMENT_DURATION = 10
SEGMENT_SAMPLES = SEGMENT_DURATION * SAMPLE_RATE
resampler = torchaudio.transforms.Resample(24000, SAMPLE_RATE)

SED_COLUMNS = ["event_label", "onset", "offset", "filename"]


def merge_with_overlap_markers(words, events):
    timeline = []

    for e in events:
        tag = SED_REVERSE_TAGS.get(e.get("event_label"))
        if tag is None:
            continue

        timeline.append((e["onset"], 0, tag))

    for w in words:
        timeline.append((w["onset"], 1, w["text"]))

    timeline.sort(key=lambda x: (float(x[0]), x[1], x[2]))

    output = []
    used = set()

    for _, typ, text in timeline:
        if typ == 0:
            if text not in used:
                output.append(f"[{text}]")
                used.add(text)
        else:
            output.append(text)
            used = set()

    return " ".join(output)


def merge(filename_suffix):
    audio_dir = Path("audio")
    word_events_folder = Path("outputs/punctuate")
    sound_events_folder = Path("outputs/sed_medmet" + filename_suffix)

    output_file = Path("outputs/text" + filename_suffix)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for audio in tqdm(list(audio_dir.rglob("*.flac"))):
            words_file = word_events_folder / audio.with_suffix(".csv").name
            sounds_file = sound_events_folder / audio.with_suffix(".csv").name

            words = (
                list(csv.DictReader(words_file.open(encoding="utf-8", newline="")))
                if words_file.exists()
                else []
            )

            sounds = (
                list(csv.DictReader(sounds_file.open(encoding="utf-8", newline="")))
                if sounds_file.exists()
                else []
            )

            merged_text = merge_with_overlap_markers(words, sounds)
            f.write(f"{audio.stem} {merged_text}\n")


def add_medmet(filename_suffix):
    medmet_dir = Path("outputs/medmet_aligner")
    sed_dir = Path("outputs/sed" + filename_suffix)

    output_dir = Path("outputs/sed_medmet" + filename_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepaths = sorted(medmet_dir.glob("*.csv"))

    for filepath in tqdm(filepaths):
        if filepath.stem == "alignment_analysis":
            continue

        utt = filepath.stem
        sed_file = sed_dir / f"{utt}.csv"
        output_file = output_dir / f"{utt}.csv"

        medmet_events = pd.read_csv(filepath)

        # handle missing SED file
        if not sed_file.exists():
            output_events = pd.DataFrame(columns=SED_COLUMNS)

        # handle empty SED file
        else:
            try:
                output_events = pd.read_csv(sed_file)
            except pd.errors.EmptyDataError:
                print(f"Empty SED file: {sed_file}")
                output_events = pd.DataFrame(columns=SED_COLUMNS)

        for col in SED_COLUMNS:
            if col not in output_events.columns:
                output_events[col] = pd.Series(dtype="object")

        medmet_mask = medmet_events["text"].isin(["e", "ee", "eee"])

        additions = medmet_events.loc[medmet_mask, ["onset", "offset"]].copy()
        additions["event_label"] = "medmet"
        additions["filename"] = "o.wav"

        output_events = pd.concat(
            [output_events[SED_COLUMNS], additions[SED_COLUMNS]],
            ignore_index=True,
        )

        output_events.to_csv(output_file, index=False)


def create_model(device, model_type):
    if model_type == "BEATS":
        beats = BEATsWrapper()
        model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")

    elif model_type == "ATST":
        atst = ATSTWrapper()
        model = PredictionsWrapper(atst, checkpoint="ATST-F_strong_1")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    model.to(device)

    return model


def load_audio(audio_file):
    waveform, sr = torchaudio.load(audio_file)

    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")

    waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len == 0:
        raise ValueError(f"Empty audio file: {audio_file}")

    return waveform, waveform_len


def load_chunks(audio_file):
    waveform, waveform_len = load_audio(audio_file)

    audio_len = waveform_len / SAMPLE_RATE

    num_chunks = (waveform_len + SEGMENT_SAMPLES - 1) // SEGMENT_SAMPLES
    padded_len = num_chunks * SEGMENT_SAMPLES

    if waveform_len < padded_len:
        waveform = torch.nn.functional.pad(waveform, (0, padded_len - waveform_len))

    chunks = waveform.view(num_chunks, SEGMENT_SAMPLES)

    return chunks, audio_len


def sound_event_detection(audio_file, device, model, threshold, median_filter):
    chunks, audio_len = load_chunks(audio_file)
    chunks = chunks.to(device, non_blocking=True)

    encoder = ManyHotEncoder(
        audioset_classes.as_strong_train_classes,
        audio_len=SEGMENT_DURATION,
    )

    with torch.inference_mode():
        mel = model.mel_forward(chunks)
        y_strong, _ = model(mel)
        y_strong = torch.sigmoid(y_strong)

    all_predictions = []

    for chunk_idx in range(y_strong.shape[0]):
        (_, _, decoded_predictions) = batched_decode_preds(
            y_strong[chunk_idx:chunk_idx + 1].float(),
            [str(audio_file)],
            encoder,
            median_filter=median_filter,
            thresholds=[threshold],
        )

        chunk_predictions = decoded_predictions[threshold].copy()

        if len(chunk_predictions) == 0:
            continue

        time_offset = chunk_idx * SEGMENT_DURATION
        chunk_predictions["onset"] += time_offset
        chunk_predictions["offset"] += time_offset

        all_predictions.append(chunk_predictions)

    # return empty dataframe with columns, not totally empty dataframe
    if not all_predictions:
        return pd.DataFrame(columns=SED_COLUMNS)

    predictions = pd.concat(all_predictions, ignore_index=True)

    predictions = predictions[predictions["onset"] < audio_len]
    predictions["offset"] = predictions["offset"].clip(upper=audio_len)

    predictions = predictions.sort_values(by="onset").reset_index(drop=True)

    # keep only expected columns if extra columns exist
    for col in SED_COLUMNS:
        if col not in predictions.columns:
            predictions[col] = ""

    return predictions[SED_COLUMNS]


def sed_inference(model_name, threshold, median_filter, filename_suffix):
    audio_dir = Path("audio")

    output_dir = Path("outputs/sed" + filename_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(device, model_name)

    filepaths = sorted(audio_dir.rglob("*.flac"))

    for filepath in tqdm(filepaths):
        try:
            predictions = sound_event_detection(
                filepath,
                device,
                model,
                threshold,
                median_filter,
            )

            output_file = output_dir / filepath.with_suffix(".csv").name

            if predictions.empty:
                predictions = pd.DataFrame(columns=SED_COLUMNS)

            predictions.to_csv(output_file, index=False)

        except Exception as e:
            print(f"Failed on {filepath}: {e}")

            output_file = output_dir / filepath.with_suffix(".csv").name
            pd.DataFrame(columns=SED_COLUMNS).to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["ATST", "BEATS"], default=SED_MODEL_NAME)  # FIX
    parser.add_argument("--median_filter", type=int, default=SED_MEDIAN_FILTER)
    parser.add_argument("--threshold", type=float, default=SED_THRESHOLD)

    args = parser.parse_args()

    model_name = args.model_name
    median_filter = args.median_filter
    threshold = args.threshold

    filename_suffix = f"_{model_name}_{threshold}_{median_filter}"

    sed_inference(model_name, threshold, median_filter, filename_suffix)
    add_medmet(filename_suffix)
    merge(filename_suffix)