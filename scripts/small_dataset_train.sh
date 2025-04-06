#!/bin/bash

export WANDB_API_KEY="75a3e7bc92541ff598fabcf666d5e781e5f84156"
# export WANDB_ENTITY="your_entity"
# export WANDB_PROJECT="text-sentiment-comparison" # Or your preferred project

GPUS="0,1,2,3" # Adjust to your setup
NUM_GPUS=$(echo $GPUS | awk -F',' '{print NF}')
MASTER_PORT=$((RANDOM % 10000 + 20000))

# This script specifically runs the RoBERTa train-from-scratch experiment
RUN_NAME="RoBERTa-TrainFromScratch-Imdb-New" # Give it a unique name

# Uses ds_config.json by default within CustomTrainer if not specified otherwise
CMD="deepspeed --include localhost:$GPUS --master_port $MASTER_PORT \
    pipeline/small_dataset.py \
    --init train \
    --run_name $RUN_NAME \
    --local_rank" # local_rank passed automatically

echo "Running RoBERTa Train-From-Scratch:"
echo $CMD
eval $CMD