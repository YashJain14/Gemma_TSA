import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

class SlicedGemma(nn.Module):
    def __init__(self, model_name="google/gemma-3-1b-pt", num_labels=2):
        super(SlicedGemma, self).__init__()

        print(f"Loading Gemma model: {model_name}")
        # Load Gemma model with output_hidden_states=True to access all layer outputs
        self.gemma = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_hidden_states=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Freeze Gemma model parameters to ensure we only train the classification layers
        for param in self.gemma.parameters():
            param.requires_grad = False

        # Get the number of layers in the model
        self.num_layers = len(self.gemma.model.layers)
        print(f"Number of layers in model: {self.num_layers}")
        
        # Get the hidden dimension size
        self.hidden_dimension = self.gemma.config.hidden_size
        print(f"Hidden dimension: {self.hidden_dimension}")
        
        self.num_labels = num_labels
        
        # Create a classification layer for each hidden layer
        # We add one more layer to include the embedding layer output
        self.classification_layers = nn.ModuleList([
            nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers + 1)
        ])
    
    def forward(self, input_ids, attention_mask=None):
        # Forward pass through Gemma model
        with torch.no_grad():
            outputs = self.gemma(input_ids, attention_mask=attention_mask)
        
        # Get all hidden states (including the embedding layer)
        all_hidden_states = outputs.hidden_states  # Shape: (num_layers+1, batch_size, sequence_length, hidden_dimension)
        
        # For classification, use the last token of each sequence
        if attention_mask is not None:
            # Find the last non-padding token position
            last_token_positions = attention_mask.sum(dim=1) - 1  # Shape: (batch_size)
        else:
            # Use the last token position if no attention mask is provided
            last_token_positions = torch.full(
                (input_ids.shape[0],), 
                input_ids.shape[1] - 1, 
                dtype=torch.long, 
                device=input_ids.device
            )
        
        # Create a range array for the batch dimension
        batch_range = torch.arange(input_ids.shape[0], device=input_ids.device)
        
        # Select the last token for each layer and batch
        all_output_logits = []
        for layer_idx, hidden_states in enumerate(all_hidden_states):
            # Extract the last token representation for each sequence in the batch
            # Shape: (batch_size, hidden_dimension)
            last_token_hidden_states = hidden_states[batch_range, last_token_positions] 
            
            # Pass through the classification layer
            logits = self.classification_layers[layer_idx](last_token_hidden_states)  # Shape: (batch_size, num_labels)
            all_output_logits.append(logits)
        
        # Stack all layer outputs
        all_output_logits_tensor = torch.stack(all_output_logits, dim=0)  # Shape: (num_layers+1, batch_size, num_labels)
        
        return all_output_logits_tensor

def compute_loss(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_labels: int, num_layers: int):
    """
    Compute loss for each layer's predictions.
    
    Args:
        all_layer_logits (torch.Tensor): Shape (num_layers+1, batch_size, num_labels)
        labels (torch.Tensor): Shape (batch_size)
        num_labels (int): Number of classes
        num_layers (int): Number of layers
    
    Returns:
        tuple: (summed_loss, all_layer_loss)
            - summed_loss: Mean loss across all layers
            - all_layer_loss: Loss for each layer
    """
    # Replicate labels for each layer
    # From shape (batch_size) to ((num_layers+1)*batch_size)
    batch_size = labels.size(0)
    repeated_labels = labels.repeat(num_layers + 1)  # +1 to include embedding layer
    
    # Reshape logits for loss calculation
    # From shape (num_layers+1, batch_size, num_labels) to ((num_layers+1)*batch_size, num_labels)
    reshaped_logits = all_layer_logits.reshape(-1, num_labels)
    
    # Calculate cross entropy loss
    all_loss = F.cross_entropy(reshaped_logits, repeated_labels, reduction='none')
    
    # Reshape loss to separate by layer
    all_layer_loss = all_loss.view(num_layers + 1, batch_size).mean(dim=1)  # Shape: (num_layers+1)
    
    # Mean loss for backpropagation
    summed_loss = all_layer_loss.mean()  # Scalar value
    
    return summed_loss, all_layer_loss