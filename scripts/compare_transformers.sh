#!/bin/bash

# --- Configuration ---
export WANDB_API_KEY="75a3e7bc92541ff598fabcf666d5e781e5f84156"
# Optional: Set entity and project if not using defaults in trainer.py
# export WANDB_ENTITY="your_entity"
# export WANDB_PROJECT="text-sentiment-comparison"

# Define GPU configuration for DeepSpeed
# Adjust based on your available GPUs
GPUS="0,1,2,3"
NUM_GPUS=$(echo $GPUS | awk -F',' '{print NF}')
MASTER_PORT=$((RANDOM % 10000 + 20000)) # Random port for multi-GPU

# --- Experiment Parameters ---
DATASET="imdb" # Options: imdb, sst2, yelp
# MODEL="roberta" # Options: roberta, gemma, llama3_2
# SUBSET_YELP="false" # Set to "true" to subset yelp
SUBSET_SIZE=25000
NUM_EPOCHS=3

# --- Loop through models ---
for MODEL in "roberta" "gemma" "llama3_2"
do
  echo "Running comparison for MODEL: $MODEL on DATASET: $DATASET"

  # Construct DeepSpeed command
  # Use ds_config.json (the general one for Trainer)
  CMD="deepspeed --include localhost:$GPUS --master_port $MASTER_PORT \
      pipeline/compare_transformers.py \
      --dataset $DATASET \
      --model $MODEL \
      --num_epochs $NUM_EPOCHS \
      --local_rank" # local_rank passed automatically by deepspeed launcher

  # Add subset arguments if needed
  if [[ "$DATASET" == "yelp" && "$SUBSET_YELP" == "true" ]]; then
    CMD+=" --subset_yelp --subset_size $SUBSET_SIZE"
  fi

  echo "Executing: $CMD"
  eval $CMD

  # Check exit status
  if [ $? -ne 0 ]; then
    echo "Error running experiment for MODEL: $MODEL. Exiting."
    exit 1
  fi

  echo "Finished experiment for MODEL: $MODEL"
  echo "--------------------------------------"

done

echo "All comparison experiments finished."