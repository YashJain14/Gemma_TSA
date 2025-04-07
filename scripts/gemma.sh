#!/bin/bash

export WANDB_API_KEY="75a3e7bc92541ff598fabcf666d5e781e5f84156"
# export WANDB_ENTITY="your_entity"
# export WANDB_PROJECT="text-sentiment-analysis-sliced"

GPUS="0,1,2,3" # Adjust to your setup
NUM_GPUS=$(echo $GPUS | awk -F',' '{print NF}')
MASTER_PORT=$((RANDOM % 10000 + 20000))

DATASET="imdb" # imdb or yelp
SUBSET_YELP="false" # true or false
SUBSET_SIZE=25000
DS_CONFIG="ds_config_gemma.json"
MODEL_NAME="google/gemma-3-1b-it" # Or gemma-7b if you have resources

CMD="deepspeed --include localhost:$GPUS --master_port $MASTER_PORT \
    pipeline/gemma.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --ds_config $DS_CONFIG \
    --local_rank" # local_rank passed automatically

if [[ "$DATASET" == "yelp" && "$SUBSET_YELP" == "true" ]]; then
  CMD+=" --subset_yelp --subset_size $SUBSET_SIZE"
fi

echo "Running Gemma Slicing:"
echo $CMD
eval $CMD