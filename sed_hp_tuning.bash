
#!/bin/bash

MODELS=("ATST" "BEATS")
THRESHOLDS=(0.05 0.1 0.3 0.4)
MEDIAN_FILTERS=(3 6 9 12)

for model in "${MODELS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
        for median_filter in "${MEDIAN_FILTERS[@]}"; do

            echo "Running: model=$model threshold=$threshold median_filter=$median_filter"

            conda run --no-capture-output -p ./envs/ptsed \
                python -u infer_sound.py \
                --model_name "$model" \
                --threshold "$threshold" \
                --median_filter "$median_filter"

        done
    done
done