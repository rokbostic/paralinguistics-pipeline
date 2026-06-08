from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


MODEL = "gemini-2.5-flash"
#MODEL = "gemini-2.5-pro"

INSTRUCTION = """
You are ANNOTATING Slovenian speech.

You are given an audio file and a transcription.
Listen to the provided audio file and annotate the given transcription with:
1. one global emotion label in curly brackets at the begginning of the text
2. paralinguistic annotations throughout the text

Return ONLY the final annotated transcription in one line and nothing else.
Do not change the given transcription in any way whatsoever. Keep it as it is even if it is wrong. Only enrich it with annotations.

EMOTION RULES:

At the very beginning of the transcription add EXACTLY ONE emotion tag inside curly brackets {}.

Choose ONLY from:
{nevtralno}
{sreča}
{žalost}
{jeza}
{drugo}

PARALINGUISTIC RULES:

Annotate paralinguistic events only when they are clearly audible. Only annotate the listed events.

For short/discrete events occurring between words use:
[smeh]
[dihanje]
[medmet]
[aplavz]

For continuous events spanning multiple words use:
<smeh> ... </smeh>
<aplavz> ... </aplavz>

Meaning of tags:
- smeh = laughter, giggling, snickering and anything similar. Use continuous form if annotating laughter that lasts through multiple words.
- dihanje = breathing in or out during speech, or panting. Only possible between words.
- medmet = the filled pause between words, the umm or eee sound.
- aplavz = applause or cheering coming from the crowd.

EXAMPLES:

Dober dan! Kako si kaj danes? Si dobro?
->
{sreča} <smeh> Dober dan </smeh>! Kako si kaj danes? Si dobro?

Gospod je poudaril, da je njihova prva odločitev za zeleno energijo pomembna tudi zaradi prihodka.
->
{nevtralno} Gospod je poudaril, [medmet] da je njihova prva odločitev za zeleno energijo [dihanje] [medmet] pomembna tudi zaradi prihodka.

Gospod Arčon, slišali smo seveda to, da največ kritik leti prav na vas
->
{jeza} [aplavz] Gospod Arčon, [dihanje] slišali smo seveda to, da največ kritik leti prav na vas 


THE TRANSCRIPTION YOU ARE TO ANNOTATE IS THE FOLLOWING:

"""

from pathlib import Path
from tqdm import tqdm
import time

import time
from pathlib import Path
from tqdm import tqdm
from google import genai

import random

# ščž

audio_models = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

def main():
    client = genai.Client(api_key=API_KEY)
    audio_dir = Path("audio")

    audio_files = list(audio_dir.rglob("*.flac"))

    text_fil = Path("outputs/gemini_text")
    emotions_fil = Path("outputs/gemini_emotions")
    
    text_fil.touch(exist_ok=True)
    emotions_fil.touch(exist_ok=True)

    transcriptions = Path("text")

    texts = dict(line.split(maxsplit=1) for line in transcriptions.read_text().splitlines())
    done_utts = {line.split(maxsplit=1)[0] for line in text_fil.read_text().splitlines()}

    for audio_fil in tqdm(audio_files):
        utt = audio_fil.stem

        if utt in done_utts:
            continue

        text = texts[utt]

        attempt = 0

        audio_file = client.files.upload(file=audio_fil)
        while True:
            try:
                response = client.models.generate_content(model= MODEL, contents=[audio_file, INSTRUCTION + text], config={"temperature": 0.0})

                response_text = response.text.strip()
                annotated_text = response_text.split("}")[1].strip()
                emotion = response_text.split("}")[0].split("{")[1]

                with open(text_fil, "a") as f:
                    f.write(f"{utt} {annotated_text}\n")
                with open(emotions_fil, "a") as f:
                    f.write(f"{utt} {emotion}\n")
                break

            except Exception as e:
                msg = str(e)
                attempt += 1

                if (
                    "429" in msg
                    or "RESOURCE_EXHAUSTED" in msg
                    or "500" in msg
                    or "INTERNAL" in msg
                    or "503" in msg
                    or "UNAVAILABLE" in msg
                    or "high demand" in msg.lower()
                ):
                    wait = min(300, 30 * attempt) + random.uniform(0, 5)
                    print(f"Temporary API issue. Retry {attempt}. Waiting {wait:.1f}s: {audio_fil}, {e}")
                    time.sleep(wait)

                else:
                    print(f"Non-retryable error with {audio_fil}: {e}")
                    break

if __name__ == "__main__":
    main()