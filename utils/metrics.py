import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

def compute_metrics(predictions, labels, num_classes=2):
    """
    Compute evaluation metrics for classification.
    
    Args:
        predictions: Model predictions (logits)
        labels: True labels
        num_classes: Number of classes in the task
        
    Returns:
        Dictionary of metrics
    """
    # Get class predictions
    preds = np.argmax(predictions, axis=1)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='macro'
    )
    
    # Calculate accuracy
    acc = accuracy_score(labels, preds)
    
    # Calculate AUROC
    try:
        if num_classes == 2:
            # Binary classification - use probability of positive class
            roc_auc = roc_auc_score(labels, predictions[:, 1])
        else:
            # Multi-class - use OvR approach
            roc_auc = roc_auc_score(
                labels, 
                predictions, 
                multi_class='ovr', 
                average='macro'
            )
    except:
        # If AUROC calculation fails (e.g., single class in split)
        roc_auc = 0.0
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": roc_auc
    }

def layer_analysis(all_metrics, best_metric='accuracy'):
    """
    Analyze performance across layers to find the best performing layer.
    
    Args:
        all_metrics: Dictionary of metrics for each layer
        best_metric: Metric to use for determining best layer
        
    Returns:
        Dictionary with best layer and its performance
    """
    # Get scores for the target metric across all layers
    metric_scores = {
        layer: metrics[best_metric] 
        for layer, metrics in all_metrics.items()
    }
    
    # Find the best layer
    best_layer = max(metric_scores.keys(), key=lambda k: metric_scores[k])
    best_score = metric_scores[best_layer]
    
    # Calculate improvement over first layer
    first_layer_score = metric_scores.get(0, 0)
    improvement = best_score - first_layer_score
    
    # Calculate improvement over last layer
    last_layer = max(metric_scores.keys())
    last_layer_score = metric_scores[last_layer]
    improvement_over_last = best_score - last_layer_score
    
    return {
        "best_layer": best_layer,
        "best_score": best_score,
        "improvement_over_first": improvement,
        "improvement_over_last": improvement_over_last,
        "all_scores": metric_scores
    }