#!/bin/bash

<<com
Usage: 
bash scripts/rjw/preprocess_dataset.sh

Helpful data preprocessing instructions here:
https://github.com/woct0rdho/ACE-Step?tab=readme-ov-file
com

PROJECT_DIR="/home/rileywilliams/soundverse/ACE-step-lowvram"
DATASET_DIR="/mnt/disks/lora_training_data/rjw_all_preprocessed"

INPUT_DATASET_DIR="$DATASET_DIR/00_converted_to_hf_dataset"
OUTPUT_DATASET_DIR="$DATASET_DIR/01_converted_to_emb_processed_dataset"

# ======================================== 
# ===== RUN ==============================
# ======================================== 
source ~/.zshrc
conda init
conda activate ace_step
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $PROJECT_DIR/preprocess_dataset_new.py \
--input_name $INPUT_DATASET_DIR \
--output_dir $OUTPUT_DATASET_DIR