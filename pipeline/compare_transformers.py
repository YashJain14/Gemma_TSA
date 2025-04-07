import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))
import argparse
import wandb
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from utils.trainer import CustomTrainer
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset

# Import PEFT modules for parameter-efficient fine-tuning
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Mapping from user-friendly names to Hugging Face model identifiers
MODEL_MAP = {
    "roberta": "roberta-base",
    "gemma": "google/gemma-2-2b-it",  # Use Gemma 2 which is supported
    "llama3_2": "meta-llama/Llama-3.2-1B"
}

def main():
    parser = argparse.ArgumentParser(description='Compare Transformer Architectures')

    parser.add_argument("--dataset", choices=['imdb', 'yelp', 'sst2'], default='imdb', help="Dataset to use")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), default='roberta', help='Model architecture to use')
    parser.add_argument("--subset_yelp", action='store_true', help='Whether to subset the yelp dataset')
    parser.add_argument("--subset_size", type=int, default=25000, help="Size for subsetting train/val/test")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenization")
    parser.add_argument("--quantize", choices=['none', '8bit', '4bit'], default='none', 
                        help="Whether to quantize the model to 8-bit or 4-bit precision")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout value")

    # for deepspeed
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    print(args)

    # --- Determine Model Name ---
    if args.model not in MODEL_MAP:
        raise ValueError(f"Unknown model key: {args.model}. Available: {list(MODEL_MAP.keys())}")
    model_name = MODEL_MAP[args.model]

    run_name = f"{args.model}-CompareTransformers-{args.dataset}"
    if args.subset_yelp:
        run_name += f"_subset{args.subset_size}"
    if args.quantize != 'none':
        run_name += f"-{args.quantize}-LoRA"

    # --- Set up Dataset ---
    input_col_name = "text" # Default
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
    elif args.dataset =='yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
        input_col_name = "text"
    elif args.dataset == 'sst2':
        dataset = load_dataset("glue", "sst2") # Correct loading for GLUE tasks
        num_labels = 2
        input_col_name = "sentence"
        # SST2 validation split is often used as test, need manual split
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['test'] = load_dataset("glue", "sst2")['validation'] # Use original validation as test
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    # --- Set up Model ---
    print(f"Loading model: {model_name}")

    # Determine if we need memory optimizations based on model size
    use_memory_optimization = args.model in ["gemma", "llama3_2"] or args.quantize != 'none'
    use_lora = args.quantize != 'none'  # Use LoRA when quantizing

    # Configure quantization if requested
    if args.quantize != 'none':
        print(f"Applying {args.quantize} quantization for {args.model}")
        if args.quantize == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
        elif args.quantize == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # Use float16 for better compatibility
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    else:
        quantization_config = None

    # Load model with quantization if applicable
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        quantization_config=quantization_config
    )

    # Apply LoRA if using quantization
    if use_lora:
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=None,  # Auto-detect target modules
            bias="none",
            inference_mode=False,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing after model is created if needed
    if use_memory_optimization and hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        
    # Disable model caching for inference to save memory
    if use_memory_optimization and hasattr(model.config, "use_cache"):
        print("Disabling model caching")
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Handle Padding Token ---
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        else:
            # Try adding a pad token
            print("Adding new pad token")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
            if model.config.pad_token_id is None: # Check if setting worked
                 raise ValueError(f"Could not set pad_token_id for {model_name}")

    # --- Set Padding Side (Crucial for Causal LMs like Gemma/Llama) ---
    if model.config.model_type in ["llama", "gemma"]:
         print(f"Setting padding side to 'left' for {model.config.model_type}")
         tokenizer.padding_side = 'left'

    # --- Prepare Dataset ---
    tokenized_datasets = tokenize(dataset, tokenizer, input_col_name=input_col_name, max_length=args.max_length)

    if args.dataset == 'sst2':
        train_dataset = tokenized_datasets["train"]
        val_dataset = tokenized_datasets["test"] # This was the 10% split from train
        test_dataset = tokenized_datasets["test"] # Original validation split
    else:
        train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    if args.dataset =='yelp' and args.subset_yelp:
        print(f"Subsetting Yelp dataset to size: {args.subset_size}")
        train_dataset = subset_dataset(train_dataset, size=args.subset_size, seed=42)
        val_dataset = subset_dataset(val_dataset, size=args.subset_size, seed=42)
        test_dataset = subset_dataset(test_dataset, size=args.subset_size, seed=42)

    # --- Create Trainer ---
    trainer = CustomTrainer(
        run_name=run_name,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        num_train_epochs=args.num_epochs
    )
    
    # Apply memory optimizations for large models
    if use_memory_optimization:
        print(f"Applying memory optimizations for {args.model}")
        
        # Set smaller batch size and more gradient accumulation steps for large models
        if args.quantize == '4bit':
            # 4-bit allows slightly larger batches
            trainer.args.per_device_train_batch_size = 2
            trainer.args.gradient_accumulation_steps = 8
        else:
            # More conservative settings for 8-bit or no quantization
            trainer.args.per_device_train_batch_size = 1
            trainer.args.gradient_accumulation_steps = 16
            
        # Use a larger batch size for evaluation since it uses less memory
        trainer.args.per_device_eval_batch_size = trainer.args.per_device_train_batch_size * 2
        
        # Disable mixed precision when using quantization to avoid compatibility issues
        if args.quantize != 'none':
            trainer.args.fp16 = False
            trainer.args.bf16 = False
    
    # Disable finding unused parameters to improve performance
    if hasattr(trainer.args, "ddp_find_unused_parameters"):
        trainer.args.ddp_find_unused_parameters = False
    
    # --- Train ---
    print(f"Starting training with batch size: {trainer.args.per_device_train_batch_size}, " 
          f"grad accum: {trainer.args.gradient_accumulation_steps}")
    trainer.train()

    # --- Evaluate on Test Set ---
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_results)
    if trainer.is_world_process_zero():
        wandb.log({"test_accuracy": test_results.get("eval_accuracy", 0.0),
                   "test_loss": test_results.get("eval_loss", 0.0)})

    print("Training finished.")
    wandb.finish()


if __name__ == "__main__":
    main()