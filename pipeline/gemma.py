import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1]))

import argparse
import json
import os

import deepspeed
import wandb
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import load_dataset
from transformers import EvalPrediction

from models.SlicedGemma import SlicedGemma, compute_loss
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset
from utils.trainer import compute_metrics

NUM_EPOCH = 3

def main():
    parser = argparse.ArgumentParser(description='Sliced Gemma model experiment')
    parser.add_argument("--dataset", choices=['imdb', 'yelp'], default='imdb', help="Dataset to use")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B", help="Model name or path")
    parser.add_argument("--subset", action="store_true", help="Whether to subset the dataset")
    parser.add_argument("--subset_size", type=int, default=25000, help="Size of dataset subset")
    
    # for deepspeed
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    print(f"Running with arguments: {args}")

    # set up dataset
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
    elif args.dataset == 'yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    print(f"Dataset {args.dataset} loaded")

    # set up model
    ds_config_path = "ds_config_gemma.json"
    if not os.path.exists(ds_config_path):
        # If gemma config doesn't exist, use llama config
        ds_config_path = "ds_config_llama.json"
    
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)
    
    model = SlicedGemma(model_name=args.model_name, num_labels=num_labels)
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # set up dataset
    tokenizer_name = args.model_name
    print(f"Tokenizing dataset with {tokenizer_name}")
    tokenized_datasets = tokenize(dataset, tokenizer_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)

    if args.subset or args.dataset == 'yelp':
        # Subset for faster training or if yelp (which is large)
        subset_size = args.subset_size
        print(f"Subsetting datasets to {subset_size} examples")
        train_dataset = subset_dataset(train_dataset, size=subset_size, seed=42)
        val_dataset = subset_dataset(val_dataset, size=subset_size//5, seed=42)
        test_dataset = subset_dataset(test_dataset, size=subset_size//5, seed=42)

    # set up dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=ds_config["train_batch_size"])
    val_dataloader = DataLoader(val_dataset, batch_size=ds_config["train_batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=ds_config["train_batch_size"])

    # Get model name for run name
    model_short_name = args.model_name.split("/")[-1]
    
    # start wandb tracking
    run_name = f"Sliced {model_short_name} - {args.dataset}"
    if args.subset:
        run_name += f" - subset {args.subset_size}"
        
    if model_engine.global_rank == 0:
        wandb.init(
            project="text-sentiment-analysis",
            name=run_name,
            config={
                "model": args.model_name,
                "dataset": args.dataset,
                "num_epochs": NUM_EPOCH,
                "batch_size": ds_config["train_batch_size"],
                "subset": args.subset,
                "subset_size": args.subset_size if args.subset else "full"
            }
        )

    for epoch in range(NUM_EPOCH):
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCH}")

        # Training loop
        model_engine.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(model_engine.device)
            attention_mask = batch['attention_mask'].to(model_engine.device)
            labels = batch['label'].to(model_engine.device)

            # forward
            all_output_logits = model_engine(input_ids=input_ids, 
                                            attention_mask=attention_mask)

            # compute loss
            summed_loss, all_layer_loss = compute_loss(
                all_layer_logits=all_output_logits, 
                labels=labels,
                num_layers=model.num_layers, 
                num_labels=model.num_labels
            )

            # backward propagation
            model_engine.backward(summed_loss)

            # weight update
            model_engine.step()

            # prepare wandb logs
            wandb_log = {"train/step": step + epoch * len(train_dataloader)}
            for i in range(model.num_layers):
                wandb_log[f"train/loss_layer_{i+1}"] = all_layer_loss[i].item()

            # Validation loop - run periodically during training
            if step % 500 == 0:
                model_engine.eval()
                
                # Collect predictions and labels for all validation batches
                all_labels = []
                all_layer_predictions = [[] for _ in range(model.num_layers)]
                
                with torch.no_grad():
                    for val_batch in tqdm(val_dataloader, desc="Validation"):
                        val_input_ids = val_batch['input_ids'].to(model_engine.device)
                        val_attention_mask = val_batch['attention_mask'].to(model_engine.device)
                        val_labels = val_batch['label'].to(model_engine.device)
                        
                        # Forward pass
                        layer_logits = model_engine(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask
                        )
                        
                        # Store predictions from each layer
                        for layer_idx in range(model.num_layers):
                            all_layer_predictions[layer_idx].append(layer_logits[layer_idx].cpu())
                        
                        # Store labels
                        all_labels.append(val_labels.cpu())
                
                # Concatenate all batches
                all_labels = torch.cat(all_labels)
                all_layer_logits = [torch.cat(layer_preds) for layer_preds in all_layer_predictions]
                
                # Compute metrics for each layer
                for layer_idx in range(model.num_layers):
                    layer_logits = all_layer_logits[layer_idx]
                    
                    # Compute loss
                    layer_loss = F.cross_entropy(layer_logits, all_labels.long())
                    wandb_log[f"val/loss_layer_{layer_idx+1}"] = layer_loss.item()
                    
                    # Compute other metrics
                    pred = EvalPrediction(
                        predictions=layer_logits.numpy(),
                        label_ids=all_labels.numpy()
                    )
                    metrics = compute_metrics(pred)
                    
                    for metric_name, value in metrics.items():
                        wandb_log[f"val/{metric_name}_layer_{layer_idx+1}"] = value
                
                # Print validation results
                if model_engine.global_rank == 0:
                    print(f"Validation - Step {step} - Epoch {epoch+1}")
                    best_layer = 0
                    best_accuracy = 0
                    for layer_idx in range(model.num_layers):
                        acc = wandb_log[f"val/accuracy_layer_{layer_idx+1}"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            best_layer = layer_idx
                    print(f"Best layer: {best_layer+1} with accuracy: {best_accuracy:.4f}")
                
                # Switch back to training mode
                model_engine.train()
                
                # Save checkpoint
                if step % 1000 == 0:
                    save_dir = f"checkpoints/{model_short_name}_{args.dataset}"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    model_engine.save_checkpoint(save_dir=save_dir)
            
            # Log to wandb
            if model_engine.global_rank == 0:
                wandb.log(wandb_log)
    
    # Final evaluation on test set
    model_engine.eval()
    
    # Collect predictions and labels for all test batches
    all_test_labels = []
    all_test_layer_predictions = [[] for _ in range(model.num_layers)]
    
    with torch.no_grad():
        for test_batch in tqdm(test_dataloader, desc="Test Evaluation"):
            test_input_ids = test_batch['input_ids'].to(model_engine.device)
            test_attention_mask = test_batch['attention_mask'].to(model_engine.device)
            test_labels = test_batch['label'].to(model_engine.device)
            
            # Forward pass
            layer_logits = model_engine(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask
            )
            
            # Store predictions from each layer
            for layer_idx in range(model.num_layers):
                all_test_layer_predictions[layer_idx].append(layer_logits[layer_idx].cpu())
            
            # Store labels
            all_test_labels.append(test_labels.cpu())
    
    # Concatenate all batches
    all_test_labels = torch.cat(all_test_labels)
    all_test_layer_logits = [torch.cat(layer_preds) for layer_preds in all_test_layer_predictions]
    
    # Compute metrics for each layer
    final_results = {}
    for layer_idx in range(model.num_layers):
        layer_logits = all_test_layer_logits[layer_idx]
        
        # Compute loss
        layer_loss = F.cross_entropy(layer_logits, all_test_labels.long())
        final_results[f"test/loss_layer_{layer_idx+1}"] = layer_loss.item()
        
        # Compute other metrics
        pred = EvalPrediction(
            predictions=layer_logits.numpy(),
            label_ids=all_test_labels.numpy()
        )
        metrics = compute_metrics(pred)
        
        for metric_name, value in metrics.items():
            final_results[f"test/{metric_name}_layer_{layer_idx+1}"] = value
    
    # Log final results
    if model_engine.global_rank == 0:
        wandb.log(final_results)
        
        # Find best layer
        best_layer = 0
        best_accuracy = 0
        for layer_idx in range(model.num_layers):
            acc = final_results[f"test/accuracy_layer_{layer_idx+1}"]
            if acc > best_accuracy:
                best_accuracy = acc
                best_layer = layer_idx
        
        # Add summary metrics
        summary = {
            "test_best_layer": best_layer + 1,
            "test_best_accuracy": best_accuracy
        }
        for key, value in summary.items():
            wandb.run.summary[key] = value
        
        print("\nTest Results Summary:")
        print(f"Best Layer: {best_layer + 1}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        # Save final model
        save_dir = f"checkpoints/{model_short_name}_{args.dataset}_final"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        model_engine.save_checkpoint(save_dir=save_dir)
        
        wandb.finish()

if __name__ == "__main__":
    main()