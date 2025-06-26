#!/bin/bash

<<com
Usage: 
bash generate_prompts_and_lyrics.sh

Helpful data preprocessing instructions here:
https://github.com/woct0rdho/ACE-Step?tab=readme-ov-file
com

PROJECT_DIR="/home/rileywilliams/soundverse/ACE-step-lowvram"
AUDIO_DIR="/mnt/disks/lora_training_data/rjw_all_preprocessed/"
# PROCESSED_DATASET_DIR="$DATASET_DIR/processed"

# ======================================== 
# ===== RUN ==============================
# ======================================== 
source ~/.zshrc
conda init
conda activate ace_step
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $PROJECT_DIR/generate_prompts_lyrics.py \
--data_dir $AUDIO_DIR 

