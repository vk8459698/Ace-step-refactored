
#!/bin/bash

<<com
Usage: 
bash scripts/rjw/inference.sh
com

# ======================================== 
# ========== LOCAL DETAILS ===============
# ======================================== 
PROJECT_DIR="/home/rileywilliams/soundverse/ACE-step-lowvram"

# ======================================== 
# ===== RUN ==============================
# ======================================== 
source ~/.zshrc
conda init
conda activate ace_step
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python $PROJECT_DIR/scripts/rjw/inference.py