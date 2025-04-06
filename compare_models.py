import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import re

def extract_model_name(path):
    """Extract model name from the directory path."""
    if 'llama' in path.lower():
        return 'Llama-2-7b'
    elif 'roberta' in path.lower():
        return 'RoBERTa-base'
    elif 'gemma' in path.lower() or 'llama-3.2' in path.lower():
        return 'Llama-3.2-1B'
    else:
        return os.path.basename(path)

def load_results(wandb_dir):
    """Load results from wandb files."""
    results = []
    
    for subdir in os.listdir(wandb_dir):
        run_dir = os.path.join(wandb_dir, subdir)
        if not os.path.isdir(run_dir):
            continue
            
        # Look for results or metrics files
        metric_files = glob.glob(os.path.join(run_dir, "*.json"))
        for file_path in metric_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Try to extract model, dataset, and metrics
                model_name = None
                dataset = None
                
                # Try to find model and dataset info
                if 'config' in data:
                    if 'model' in data['config']:
                        model_name = data['config']['model']
                    elif 'name' in data['config']:
                        model_name = extract_model_name(data['config']['name'])
                        
                    if 'dataset' in data['config']:
                        dataset = data['config']['dataset']
                
                # Extract metrics
                metrics = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)) and any(metric in key for metric in ['accuracy', 'f1', 'precision', 'recall']):
                        # Extract layer number if present
                        layer_match = re.search(r'layer_(\d+)', key)
                        if layer_match:
                            layer = int(layer_match.group(1))
                            metric_name = key.split('/')[0]
                            if metric_name not in metrics:
                                metrics[metric_name] = {}
                            metrics[metric_name][layer] = value
                
                # Add to results
                if model_name and dataset and metrics:
                    for metric_name, layer_values in metrics.items():
                        # Find the best layer
                        best_layer = max(layer_values.items(), key=lambda x: x[1])
                        results.append({
                            'model': model_name,
                            'dataset': dataset,
                            'metric': metric_name,
                            'best_layer': best_layer[0],
                            'best_value': best_layer[1],
                            'all_layer_values': layer_values
                        })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return results

def plot_layer_performance(results, metric='accuracy', save_dir=None):
    """Plot layer performance for different models."""
    # Filter by metric
    filtered_results = [r for r in results if r['metric'] == metric]
    
    if not filtered_results:
        print(f"No results found for metric {metric}")
        return
    
    # Group by model and dataset
    model_dataset_groups = {}
    for result in filtered_results:
        key = (result['model'], result['dataset'])
        if key not in model_dataset_groups:
            model_dataset_groups[key] = []
        model_dataset_groups[key].append(result)
    
    # Create plots
    for (model, dataset), group in model_dataset_groups.items():
        if not group:
            continue
            
        # Use the first item in the group
        result = group[0]
        if 'all_layer_values' not in result:
            continue
            
        # Create dataframe for plotting
        layers = list(result['all_layer_values'].keys())
        values = list(result['all_layer_values'].values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, values, marker='o', linestyle='-')
        plt.axhline(y=max(values), color='r', linestyle='--', alpha=0.3)
        
        plt.title(f"{model} - {dataset} - {metric.capitalize()} by Layer")
        plt.xlabel('Layer')
        plt.ylabel(metric.capitalize())
        plt.grid(True, alpha=0.3)
        
        # Add peak marker
        best_layer = max(result['all_layer_values'].items(), key=lambda x: x[1])
        plt.scatter(best_layer[0], best_layer[1], color='red', s=100, 
                    label=f'Best: Layer {best_layer[0]} ({best_layer[1]:.4f})')
        
        plt.legend()
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{model}_{dataset}_{metric}_layer_performance.png"))
            plt.close()
        else:
            plt.show()

def compare_models(results, metric='accuracy', save_dir=None):
    """Compare the best performance of different models."""
    # Filter by metric
    filtered_results = [r for r in results if r['metric'] == metric]
    
    if not filtered_results:
        print(f"No results found for metric {metric}")
        # Group by dataset
    dataset_groups = {}
    for result in filtered_results:
        dataset = result['dataset']
        if dataset not in dataset_groups:
            dataset_groups[dataset] = []
        dataset_groups[dataset].append(result)
    
    # Create plots for each dataset
    for dataset, group in dataset_groups.items():
        # Extract model names and best values
        models = []
        best_values = []
        best_layers = []
        
        for result in group:
            models.append(result['model'])
            best_values.append(result['best_value'])
            best_layers.append(result['best_layer'])
        
        # Sort by best value in descending order
        sorted_indices = sorted(range(len(best_values)), key=lambda i: best_values[i], reverse=True)
        models = [models[i] for i in sorted_indices]
        best_values = [best_values[i] for i in sorted_indices]
        best_layers = [best_layers[i] for i in sorted_indices]
        
        # Create dataframe for plotting
        df = pd.DataFrame({
            'Model': models,
            f'Best {metric.capitalize()}': best_values,
            'Best Layer': best_layers
        })
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df['Model'], df[f'Best {metric.capitalize()}'], color='skyblue')
        
        # Add value labels on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            layer = df['Best Layer'].iloc[i]
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}\n(Layer {layer})',
                    ha='center', va='bottom', rotation=0)
        
        plt.title(f'Comparison of Model Performance on {dataset} ({metric.capitalize()})')
        plt.xlabel('Model')
        plt.ylabel(f'Best {metric.capitalize()}')
        plt.ylim(0, max(best_values) * 1.15)  # Add some space for the text
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{dataset}_{metric}_model_comparison.png"))
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare model performance')
    parser.add_argument('--wandb_dir', type=str, required=True, help='Directory containing wandb files')
    parser.add_argument('--save_dir', type=str, default='comparison_plots', help='Directory to save plots')
    parser.add_argument('--metrics', nargs='+', default=['accuracy', 'f1'], help='Metrics to plot')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.wandb_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Create plots
    print(f"Found {len(results)} result entries")
    for metric in args.metrics:
        print(f"Plotting {metric} performance...")
        plot_layer_performance(results, metric=metric, save_dir=args.save_dir)
        compare_models(results, metric=metric, save_dir=args.save_dir)
    
    print(f"Plots saved to {args.save_dir}")

if __name__ == "__main__":
    main()