# Get medmet
.venv/bin/python get_medmet.py

# Align captions
.venv/bin/python create_corpus.py
conda run --no-capture-output -p ./envs/mfa python -u align.py

# Get sound events
conda run --no-capture-output -p ./envs/ptsed python -u infer_sound.py

# Get emotions
conda run -p ./envs/emotion python get_emotion.py

.venv/bin/python punctuate.py
.venv/bin/python add_medmet.py
.venv/bin/python merge.py
