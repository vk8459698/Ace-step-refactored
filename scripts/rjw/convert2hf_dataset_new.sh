#!/bin/bash

<<com
Usage: 
bash scripts/rjw/convert2hf_dataset_new.sh

Helpful data preprocessing instructions here:
https://github.com/woct0rdho/ACE-Step?tab=readme-ov-file
com

PROJECT_DIR="/home/rileywilliams/soundverse/ACE-step-lowvram"
DATASET_DIR="/mnt/disks/lora_training_data/rjw_all_preprocessed"
PROCESSED_DATASET_DIR="$DATASET_DIR/00_converted_to_hf_dataset"
# PROCESSED_DATASET_DIR="$DATASET_DIR/01_converted_to_emb_processed_dataset"

# ======================================== 
# ===== RUN ==============================
# ======================================== 
source ~/.zshrc
conda init
conda activate ace_step
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $PROJECT_DIR/convert2hf_dataset_new.py \
--data_dir $DATASET_DIR \
--output_name $PROCESSED_DATASET_DIR