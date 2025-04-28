#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

for SUB_ID in {1..10}
do
    SUBJECTS=$(printf "sub-%02d" $SUB_ID)
    echo "Training subject ${SUB_ID}..."
    python main_joint.py \
        --config configs/baseline-nonorm.yaml \
        --subjects "$SUBJECTS" \
        --seed 0 \
        --exp_setting intra-subject \
        --brain_backbone "EEGProjectLayer" \
        --vision_backbone "RN50" \
        --epoch 50 \
        --lr 1e-4;
done

echo "Running average results calculating script..."
python average.py \
    --runs_dir "./exp/intra-subject_baseline-nonorm_EEGProjectLayer_RN50";