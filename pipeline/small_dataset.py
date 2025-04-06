import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))
import argparse
import wandb
from transformers import AutoModelForSequenceClassification, RobertaConfig, AutoTokenizer
from datasets import load_dataset
from utils.trainer import CustomTrainer, training_args
from utils.preprocessing import tokenize, train_val_test_split

def main():
    parser = argparse.ArgumentParser(description='Small dataset experiments (RoBERTa Focus)')

    parser.add_argument("--init", choices=['train', 'finetune'], default='train', help="Whether to train from scratch or finetune RoBERTa")
    parser.add_argument("--run_name", type=str, required=True, help="Run name for the wandb experiment")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override default epochs (20 for train, 3 for finetune)")

    # for deepspeed
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")

    args = parser.parse_args()

    print(args)

    # --- Set up Model ---
    if args.init == 'train':
        model_name = "roberta-base"
        print("Training RoBERTa from scratch...")
        config = RobertaConfig.from_pretrained(model_name, num_labels=2) # Assuming IMDB (2 labels)
        model = AutoModelForSequenceClassification.from_config(config)
        # Use more epochs for training from scratch
        num_epochs = args.num_epochs if args.num_epochs is not None else 20
    elif args.init == 'finetune':
        # Option 1: Finetune a base model
        model_name = "roberta-base"
        # Option 2: Finetune a sentiment-specific model (like original paper)
        # model_name = "siebert/sentiment-roberta-large-english" # Requires more GPU memory
        print(f"Finetuning RoBERTa: {model_name}...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        num_epochs = args.num_epochs if args.num_epochs is not None else 3
    else:
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: # Handle potential missing pad token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id


    # --- Set up Dataset (IMDB) ---
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    tokenized_datasets = tokenize(dataset, tokenizer, input_col_name="text")
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)


    # --- Create Trainer ---
    # Modify default args for this experiment
    current_training_args = training_args
    current_training_args.num_train_epochs = num_epochs
    current_training_args.run_name = args.run_name # Set run name

    trainer = CustomTrainer(
        trainer_args=current_training_args, # Use potentially modified args
        # run_name=args.run_name, # run_name is now set via trainer_args
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # --- Train ---
    print(f"Starting training for {num_epochs} epochs...")
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