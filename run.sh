# Create corpus
python create_corpus.py

# Align captions
conda run -p ./envs/mfa python align.py

conda run -p ./envs/mfa python textgrid2csv.py
conda run -p ./envs/mfa python punctuate.py

# Get sound events
conda run -p ./envs/ptsed python infer_sound.py
python merge.py

# Get emotions
conda run -p ./envs/emotion python get_emotion.py

