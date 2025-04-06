#!/bin/bash

# Create a SLURM job script for Gemma model training
cat > gemma_job.sh << EOL
#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=360
#SBATCH --job-name=gemma_tsa
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Make sure SLURM is loaded
module load slurm

# Load conda
module load anaconda

# Create or activate conda environment
if conda env list | grep -q "gemma_env"; then
    echo "Activating existing gemma_env environment"
    source activate gemma_env
else
    echo "Creating new gemma_env environment"
    conda create -y -n gemma_env python=3.9
    source activate gemma_env
    
    # Install necessary packages
    pip install torch torchvision torchaudio
    pip install transformers datasets wandb deepspeed tqdm scikit-learn importlib_metadata
fi

# Make sure deepspeed configuration exists
if [ ! -f "ds_config_gemma.json" ]; then
    echo "Copying ds_config_llama.json to ds_config_gemma.json"
    cp ds_config_llama.json ds_config_gemma.json
fi

# Run the training
python pipeline/gemma.py --dataset ${DATASET} --model_name ${MODEL_NAME} ${SUBSET_FLAG}
EOL

# Make the script executable
chmod +x gemma_job.sh

# Create a convenience script to submit the job
cat > run_gemma.sh << EOL
#!/bin/bash

# Default parameters
DATASET="imdb"
MODEL_NAME="meta-llama/Llama-3.2-1B"
SUBSET_FLAG=""

# Process command-line arguments
while [[ \$# -gt 0 ]]; do
    case \$1 in
        --dataset)
            DATASET="\$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="\$2"
            shift 2
            ;;
        --subset)
            SUBSET_FLAG="--subset"
            shift
            ;;
        --subset_size)
            SUBSET_FLAG="--subset --subset_size \$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: \$1"
            exit 1
            ;;
    esac
done

echo "Running with dataset: \${DATASET}, model: \${MODEL_NAME}"
if [ -n "\${SUBSET_FLAG}" ]; then
    echo "Using subset of data"
fi

# Submit the job
export DATASET MODEL_NAME SUBSET_FLAG
sbatch gemma_job.sh
EOL

# Make the script executable
chmod +x run_gemma.sh

# For direct DeepSpeed run without SLURM
cat > run_deepspeed_gemma.sh << EOL
#!/bin/bash

# Set your WANDB API key
WANDB_API_KEY=your_wandb_api_key

# Default parameters
DATASET="imdb"
MODEL_NAME="meta-llama/Llama-3.2-1B"
SUBSET_FLAG=""
GPU_LIST="0,1,2,3"

# Process command-line arguments
while [[ \$# -gt 0 ]]; do
    case \$1 in
        --dataset)
            DATASET="\$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="\$2"
            shift 2
            ;;
        --subset)
            SUBSET_FLAG="--subset"
            shift
            ;;
        --subset_size)
            SUBSET_FLAG="--subset --subset_size \$2"
            shift 2
            ;;
        --gpus)
            GPU_LIST="\$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: \$1"
            exit 1
            ;;
    esac
done

echo "Running with dataset: \${DATASET}, model: \${MODEL_NAME}, GPUs: \${GPU_LIST}"
if [ -n "\${SUBSET_FLAG}" ]; then
    echo "Using subset of data"
fi

# Run with DeepSpeed
WANDB_API_KEY=\${WANDB_API_KEY} \\
    deepspeed --include localhost:\${GPU_LIST} \\
    pipeline/gemma.py \\
    --dataset \${DATASET} \\
    --model_name \${MODEL_NAME} \\
    \${SUBSET_FLAG}
EOL

# Make the script executable
chmod +x run_deepspeed_gemma.sh

echo "Scripts created:"
echo "1. gemma_job.sh - SLURM job script"
echo "2. run_gemma.sh - Convenience script to submit SLURM job"
echo "3. run_deepspeed_gemma.sh - Script to run directly with DeepSpeed"
echo ""
echo "To run with SLURM:"
echo "./run_gemma.sh --dataset imdb --model_name meta-llama/Llama-3.2-1B [--subset] [--subset_size 10000]"
echo ""
echo "To run directly with DeepSpeed:"
echo "./run_deepspeed_gemma.sh --dataset imdb --model_name meta-llama/Llama-3.2-1B --gpus 0,1 [--subset] [--subset_size 10000]"