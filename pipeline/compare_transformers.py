import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))
import argparse
import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from utils.trainer import CustomTrainer
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset

# Mapping from user-friendly names to Hugging Face model identifiers
MODEL_MAP = {
    "roberta": "roberta-base",
    "gemma": "google/gemma-2b", # Using 2b as 3.1b is PT only
    "llama3_2": "meta-llama/Llama-3.2-8B-Instruct" # Using 8B instruct as 1B is not available
    # Add other models here if needed
}
# Note: For Llama/Gemma classification, tokenizer padding side might be important
# If using standard AutoModelForSequenceClassification, it often adds a head
# that pools outputs. If it relies on the *last* token, left padding is needed.

def main():
    parser = argparse.ArgumentParser(description='Compare Transformer Architectures')

    parser.add_argument("--dataset", choices=['imdb', 'yelp', 'sst2'], default='imdb', help="Dataset to use")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), default='roberta', help='Model architecture to use')
    parser.add_argument("--subset_yelp", action='store_true', help='Whether to subset the yelp dataset')
    parser.add_argument("--subset_size", type=int, default=25000, help="Size for subsetting train/val/test")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")

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
        raise NotImplementedError

    # --- Set up Model ---
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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
    # AutoModelForSequenceClassification *might* handle this, but explicit is safer
    # If the classification head uses the last token state, left padding is essential.
    if model.config.model_type in ["llama", "gemma"]:
         print(f"Setting padding side to 'left' for {model.config.model_type}")
         tokenizer.padding_side = 'left'


    # --- Prepare Dataset ---
    # Pass tokenizer explicitly to tokenize function
    tokenized_datasets = tokenize(dataset, tokenizer, input_col_name=input_col_name)

    # Adjust split function if dataset doesn't have 'test' or 'validation'
    if args.dataset == 'sst2':
        # We already split 'train' into train/validation and have 'test'
         train_dataset = tokenized_datasets["train"]
         val_dataset = tokenized_datasets["test"] # This was the 10% split from train
         test_dataset = tokenized_datasets["test"] # Original validation split
         # TODO: Re-tokenize the original validation split if needed
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
        tokenizer=tokenizer, # Pass tokenizer to trainer
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        num_train_epochs=args.num_epochs # Pass epochs from args
    )

    # --- Train ---
    print("Starting training...")
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