#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                   # Memory for RoBERTa slicing
#SBATCH --time=05:59:00
#SBATCH --job-name=slc_rob_imdb     # Specific job name
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

# --- Environment Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job Name: ${SLURM_JOB_NAME}" ; echo "Job ID: ${SLURM_JOBID}"
mkdir -p slurm_out
module purge ; module load anaconda ; source activate llm_env # Activate env
export WANDB_API_KEY="75a3e7bc92541ff598fabcf666d5e781e5f84156" # !! EDIT THIS !!
# export WANDB_PROJECT="text-sentiment-analysis-sliced" # Optional
cd "${SLURM_SUBMIT_DIR}"

# --- Experiment Specifics ---
EXPERIMENT_TYPE="slice"
MODEL_KEY="roberta"
DATASET="imdb"
PYTHON_SCRIPT="pipeline/sliced_roberta.py"
MODEL_HF_ID="roberta-base"
DS_CONFIG_FILE="ds_config_roberta.json"
EXTRA_ARGS=" --model_name $MODEL_HF_ID --dataset $DATASET --ds_config $DS_CONFIG_FILE"

# --- Construct and Run DeepSpeed Command ---
CMD="deepspeed --num_gpus=1 ${PYTHON_SCRIPT} ${EXTRA_ARGS} --local_rank=-1"
echo "Running Command: $CMD"
eval $CMD
EXIT_CODE=$?
echo "Job finished with exit code $EXIT_CODE at $(date)"
exit $EXIT_CODE