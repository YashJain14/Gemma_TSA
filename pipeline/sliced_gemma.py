import sys
import os
import argparse
import json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import deepspeed
from datasets import load_dataset

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.SlicedGemma import SlicedGemma, compute_loss
from utils.preprocessing import tokenize, train_val_test_split, subset_dataset
from utils.metrics import compute_metrics, layer_analysis

NUM_EPOCHS = 3

def main():
    parser = argparse.ArgumentParser(description='Sliced Gemma model experiment')
    parser.add_argument("--dataset", choices=['imdb', 'yelp', 'sst2'], default='imdb', help="Dataset to use")
    parser.add_argument("--subset", action='store_true', help="Whether to subset the dataset")
    parser.add_argument("--subset_size", type=int, default=25000, help="Size of the dataset subset")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-1b-pt", help="Gemma model name/path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save results and checkpoints")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting experiment with:\n- Dataset: {args.dataset}\n- Model: {args.model_name}")
    
    # Set up dataset
    if args.dataset == 'imdb':
        dataset = load_dataset("imdb")
        num_labels = 2
        input_col_name = "text"
    elif args.dataset == 'yelp':
        dataset = load_dataset("yelp_review_full")
        num_labels = 5
        input_col_name = "text"
    elif args.dataset == 'sst2':
        dataset = load_dataset("sst2")
        num_labels = 2
        input_col_name = "sentence"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Set up model with DeepSpeed
    with open(args.ds_config, "r") as f:
        ds_config = json.load(f)
    
    model = SlicedGemma(model_name=args.model_name, num_labels=num_labels)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # Prepare dataset
    tokenized_datasets = tokenize(dataset, args.model_name, input_col_name=input_col_name)
    train_dataset, val_dataset, test_dataset = train_val_test_split(tokenized_datasets)
    
    # Apply subsetting if requested
    if args.subset:
        print(f"Subsetting datasets to {args.subset_size} examples")
        train_dataset = subset_dataset(train_dataset, size=args.subset_size)
        val_dataset = subset_dataset(val_dataset, size=args.subset_size // 5)
        test_dataset = subset_dataset(test_dataset, size=args.subset_size // 5)
    
    # Set up data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=ds_config["train_batch_size"], 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=ds_config["train_batch_size"]
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=ds_config["train_batch_size"]
    )
    
    # Initialize wandb for experiment tracking
    run_name = f"SlicedGemma-{args.dataset}"
    if args.subset:
        run_name += "-subset"
    
    if model_engine.global_rank == 0:
        wandb.init(
            project="sliced-gemma-experiment",
            name=run_name,
            config={
                "model_name": args.model_name,
                "dataset": args.dataset,
                "subset": args.subset,
                "num_epochs": NUM_EPOCHS,
                "batch_size": ds_config["train_batch_size"],
                "num_layers": model.num_layers,
                "num_labels": num_labels
            }
        )
    
    # Training loop
    best_val_metrics = {}
    for epoch in range(NUM_EPOCHS):
        if model_engine.global_rank == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Training
        model_engine.train()
        train_progress = tqdm(
            train_dataloader, 
            desc=f"Training Epoch {epoch+1}", 
            disable=model_engine.global_rank != 0
        )
        
        for step, batch in enumerate(train_progress):
            # Move batch to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # Forward pass
            all_output_logits = model_engine(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # Compute loss
            summed_loss, all_layer_loss = compute_loss(
                all_layer_logits=all_output_logits, 
                labels=batch['label'],
                num_labels=model.num_labels, 
                num_layers=model.num_layers
            )
            
            # Backward pass
            model_engine.backward(summed_loss)
            
            # Update weights
            model_engine.step()
            
            # Log training metrics
            if model_engine.global_rank == 0:
                wandb_log = {"train/step": step + epoch * len(train_dataloader)}
                for i in range(model.num_layers + 1):
                    wandb_log[f"train/loss_layer_{i}"] = all_layer_loss[i].item()
                wandb.log(wandb_log)
            
            # Run validation periodically
            if step % 100 == 0:
                # Validation
                val_metrics = run_evaluation(
                    model_engine=model_engine,
                    dataloader=val_dataloader,
                    num_labels=num_labels,
                    desc="Validation"
                )
                
                # Save best metrics for each layer
                for layer, metrics in val_metrics.items():
                    if layer not in best_val_metrics or metrics["accuracy"] > best_val_metrics[layer]["accuracy"]:
                        best_val_metrics[layer] = metrics
                
                # Log validation metrics
                if model_engine.global_rank == 0:
                    wandb_log = {"val/step": step + epoch * len(train_dataloader)}
                    
                    for layer, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            wandb_log[f"val/{metric_name}_layer_{layer}"] = value
                    
                    # Add analysis
                    analysis = layer_analysis(val_metrics)
                    for key, value in analysis.items():
                        if key != "all_scores":  # Skip logging all scores again
                            wandb_log[f"analysis/{key}"] = value
                    
                    wandb.log(wandb_log)
                
                # Save checkpoint
                if step % 500 == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}-step{step}")
                    model_engine.save_checkpoint(checkpoint_dir)
        
        # End of epoch validation
        val_metrics = run_evaluation(
            model_engine=model_engine,
            dataloader=val_dataloader,
            num_labels=num_labels,
            desc=f"End of Epoch {epoch+1} Validation"
        )
        
        # Log end of epoch validation metrics
        if model_engine.global_rank == 0:
            wandb_log = {"epoch": epoch + 1}
            
            for layer, metrics in val_metrics.items():
                for metric_name, value in metrics.items():
                    wandb_log[f"epoch_val/{metric_name}_layer_{layer}"] = value
            
            # Add analysis
            analysis = layer_analysis(val_metrics)
            for key, value in analysis.items():
                if key != "all_scores":
                    wandb_log[f"epoch_analysis/{key}"] = value
            
            wandb.log(wandb_log)
    
    # Final evaluation on test set
    test_metrics = run_evaluation(
        model_engine=model_engine,
        dataloader=test_dataloader,
        num_labels=num_labels,
        desc="Test Evaluation"
    )
    
    # Save final model
    final_checkpoint_dir = os.path.join(args.output_dir, "final-model")
    model_engine.save_checkpoint(final_checkpoint_dir)
    
    # Log final test metrics
    if model_engine.global_rank == 0:
        wandb_log = {}
        
        for layer, metrics in test_metrics.items():
            for metric_name, value in metrics.items():
                wandb_log[f"test/{metric_name}_layer_{layer}"] = value
        
        # Add analysis
        analysis = layer_analysis(test_metrics)
        for key, value in analysis.items():
            if key != "all_scores":
                wandb_log[f"test_analysis/{key}"] = value
        
        wandb.log(wandb_log)
        
        # Create a summary report
        summary = {
            "test_best_layer": analysis["best_layer"],
            "test_best_accuracy": analysis["best_score"],
            "improvement_over_first_layer": analysis["improvement_over_first"],
            "improvement_over_last_layer": analysis["improvement_over_last"]
        }
        
        for key, value in summary.items():
            wandb.run.summary[key] = value
        
        print("\nTest Results Summary:")
        print(f"Best Layer: {analysis['best_layer']}")
        print(f"Best Accuracy: {analysis['best_score']:.4f}")
        print(f"Improvement over first layer: {analysis['improvement_over_first']:.4f}")
        print(f"Improvement over last layer: {analysis['improvement_over_last']:.4f}")
        
        # Save layer-wise metrics to file
        with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
            json.dump({f"layer_{layer}": metrics for layer, metrics in test_metrics.items()}, f, indent=2)
        
        # Finish wandb run
        wandb.finish()

def run_evaluation(model_engine, dataloader, num_labels, desc="Evaluation"):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model_engine: DeepSpeed model engine
        dataloader: DataLoader for evaluation
        num_labels: Number of classes
        desc: Description for the progress bar
        
    Returns:
        Dictionary of metrics for each layer
    """
    model_engine.eval()
    
    # Dictionary to store predictions and labels for each layer
    all_preds = {i: [] for i in range(model_engine.module.num_layers + 1)}
    all_labels = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, disable=model_engine.global_rank != 0):
            # Move batch to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Store predictions for each layer
            for i in range(model_engine.module.num_layers + 1):
                all_preds[i].extend(outputs[i].cpu().numpy())
            
            # Store labels
            all_labels.extend(batch['label'].cpu().numpy())
    
    # Compute metrics for each layer
    metrics = {}
    for layer, preds in all_preds.items():
        metrics[layer] = compute_metrics(
            np.array(preds),
            np.array(all_labels),
            num_classes=num_labels
        )
    
    return metrics

if __name__ == "__main__":
    main()