# Get medmet
.venv/bin/python get_medmet.py

# Align captions
.venv/bin/python create_corpus.py
conda run --no-capture-output -p ./envs/mfa python -u align.py
.venv/bin/python punctuate.py

# Get sound events (change MODEL_NAME value to "ASTS" or "BEATS" to change model)
conda run --no-capture-output -p ./envs/ptsed python -u infer_sound.py

# Get emotions
conda run --no-capture-output -p ./envs/emotion python get_emotion.py

