
#!/bin/bash

<<com
Usage: 
bash scripts/rjw/train_lora.sh

Helpful training resources here:
https://github.com/woct0rdho/ACE-Step?tab=readme-ov-file
com


# Tesla T4 does not support bfloat16 compilation natively, skipping


# ======================================== 
# ===== EXPERIMENT DETAILS ===============
# ======================================== 
PROJECT_DIR="/home/rileywilliams/soundverse/ACE-step-lowvram"
# DATASET_DIR="/mnt/disks/audio-ai-research-speech-data/ace_step_datasets/rjw_all_LoRa_data"
# PROCESSED_DATASET_DIR="$DATASET_DIR/processed"
DATASET_DIR="/mnt/disks/lora_training_data/rjw_all_preprocessed"
PROCESSED_DATASET_DIR="$DATASET_DIR/01_converted_to_emb_processed_dataset"
CONFIG="./config/lora_config_transformer_only.json"
# CONFIG="./config/lora_configy.json" # training the lyrics decoder is needed only when adding a new language
LAST_LORA_PATH="None"
EXP_NAME="ace_step_lora_rjwpersonal"

# ======================================== 
# ===== TRAINING DETAILS =================
# ======================================== 
N_EPOCHS=-1 # 3 # -1 # 1
PRECISION="bf16-true" # NEXT TRY THE DEFAULT: "bf16-mixed"#16
# PRECISION="bf16-mixed" 
# PRECISION="32-true"
# https://lightning.ai/docs/pytorch/stable/common/trainer.html

# ========================================
# ===== TRAINING INFRA ===================
# ======================================== 
N_GPUS=1
BATCH_SIZE=1
NUM_WORKERS=0
# LOGGER_DIR="${PROJECT_DIR}/loradata/exp/logs/"
CHECKPOINT_PATH=None
LORA_CHECKPOINT_DIR="${PROJECT_DIR}/models/exp/${EXP_NAME}/"
# CHECKPOINT_DIR=
LIMIT_VAL_BATCHES=0.0
SAVE_EVERY_N_TRAIN_STEPS=100 # 2000
MAX_STEPS=1000000 # 30 # 120
SAVE_LAST_N_CHECKPOINTS=5


# ======================================== 
# ===== RUN ==============================
# ======================================== 
source ~/.zshrc
conda init
conda activate ace_step
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --checkpoint_dir $CHECKPOINT_DIR \
# --last_lora_path $LAST_LORA_PATH \

python $PROJECT_DIR/trainer_new.py \
--lora_config_path $CONFIG \
--dataset_path $PROCESSED_DATASET_DIR \
--last_lora_path $LAST_LORA_PATH \
--lora_checkpoint_dir $LORA_CHECKPOINT_DIR \
--batch_size $BATCH_SIZE \
--num_workers $NUM_WORKERS \
--epochs $N_EPOCHS \
--max_steps $MAX_STEPS \
--exp_name $EXP_NAME \
--precision $PRECISION \
--save_every_n_train_steps $SAVE_EVERY_N_TRAIN_STEPS \
--save_last $SAVE_LAST_N_CHECKPOINTS

# python $PROJECT_DIR/trainer_new.py \
# --lora_config_path $CONFIG \
# --dataset_path $PROCESSED_DATASET_DIR \
# --lora_checkpoint_dir $LORA_CHECKPOINT_DIR