import argparse
import librosa
import torch
import torchaudio

from helpers import audioset_classes
from helpers.decode import batched_decode_preds
from helpers.encode import ManyHotEncoder
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
from models.m2d.M2D_wrapper import M2DWrapper
from models.asit.ASIT_wrapper import ASiTWrapper
from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
from models.prediction_wrapper import PredictionsWrapper
from models.frame_mn.utils import NAME_TO_WIDTH

from pathlib import Path
import pandas as pd

MODEL_NAME = "ATST"

def create_model(device, model_type=MODEL_NAME):

    if model_type == "BEATS":
        beats = BEATsWrapper()
        model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")

    if model_type == "ATST":
        atst = ATSTWrapper()
        model = PredictionsWrapper(atst, checkpoint="ATST-F_strong_1")

    model.eval()
    model.to(device)

    return model

SAMPLE_RATE= 16000  # all our models are trained on 16 kHz audio
SEGMENT_DURATION = 10  # all models are trained on 10-second pieces
SEGMENT_SAMPLES = SEGMENT_DURATION * SAMPLE_RATE
resampler = torchaudio.transforms.Resample(24000, SAMPLE_RATE)

def load_audio(audio_file):

    # load audio
    waveform, sr = torchaudio.load(audio_file)

    # sr    
    if sr != 24000:
        raise ValueError(f"Expected 24kHz, got {sr}")
    waveform = resampler(waveform)
    
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len == 0:
        raise ValueError(f"Empty audio file: {audio_file}")
    
    return waveform, waveform_len

def load_chunks(audio_file):

    waveform, waveform_len = load_audio(audio_file)

    audio_len = waveform_len / SAMPLE_RATE  # in seconds

    num_chunks = (waveform_len + SEGMENT_SAMPLES - 1) // SEGMENT_SAMPLES # amount of chunks needed (ceil)
    padded_len = num_chunks * SEGMENT_SAMPLES # pad to 10*n seconds
    
    if waveform_len < padded_len:
        waveform = torch.nn.functional.pad(waveform, (0, padded_len - waveform_len))

    chunks = waveform.view(num_chunks, SEGMENT_SAMPLES)
    return chunks, audio_len


def sound_event_detection(audio_file, device, model, output_dir):
    
    chunks, audio_len = load_chunks(audio_file)
    chunks = chunks.to(device, non_blocking=True)

    encoder = ManyHotEncoder(
        audioset_classes.as_strong_train_classes,
        audio_len=audio_len
    )

    with torch.inference_mode():
        mel = model.mel_forward(chunks)
        y_strong, _ = model(mel)
        y_strong = torch.sigmoid(y_strong)

    threshold = 0.1

    (
        scores_unprocessed,
        scores_postprocessed,
        decoded_predictions
    ) = batched_decode_preds(
        y_strong.float(),
        str(audio_file),
        encoder,
        median_filter=9,
        thresholds=[threshold],
    )

    predictions = decoded_predictions[threshold].sort_values(by="onset")   

    output_file = output_dir / audio_file.with_suffix(".csv").name
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    #print(f"Saved to: {output_file}")


def main():
    corpus_dir = Path("outputs/corpus")

    output_dir = Path("outputs/sed_" + MODEL_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model(device, MODEL_NAME)
        
    for filepath in corpus_dir.rglob("*.flac"):
        try:
            if (output_dir / filepath.with_suffix(".csv").name).exists():
                continue

            sound_event_detection(filepath, device, model, output_dir)
        except Exception as e:
            print(f"Failed on {filepath}: {e}")


if __name__ == "__main__":
    main()