import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GemmaModel, AutoConfig

class SlicedGemma(nn.Module):
    def __init__(self, model_name: str = "google/gemma-3-1b-it", num_labels: int = 2):
        super(SlicedGemma, self).__init__()

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.gemma = GemmaModel.from_pretrained(model_name, config=self.config)

        # freeze gemma model parameters
        for param in self.gemma.parameters():
            param.requires_grad = False

        self.num_layers = self.config.num_hidden_layers
        self.hidden_dimension = self.config.hidden_size
        self.num_labels = num_labels
        # Add 1 for the embedding layer? Gemma's output_hidden_states includes embeddings
        # Let's stick to just transformer layers for now. If embeddings are needed, adjust num_layers and indexing.
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.gemma(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # hidden_states tuple: (embedding_output, layer_1_output, ..., layer_n_output)
        # We want outputs from layer_1 to layer_n
        all_hidden_states = outputs.hidden_states[1:] # Exclude embedding layer

        # Convert all_hidden_states from tuple to tensor
        all_hidden_states_tensor = torch.stack(all_hidden_states, dim=0) # (num_layers, batch_size, sequence_length, hidden_dimension)

        ############# Use the last token of each sequence for classification #############
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            # Find the index of the last non-padding token
            # Sum attention mask to get sequence lengths
            sequence_lengths = torch.sum(attention_mask, dim=1) - 1 # Indices are 0-based
            last_token_indices = sequence_lengths.long() # (batch_size)
        else:
            # Assume no padding if attention_mask is None
            last_token_indices = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=input_ids.device)

        # Create indices for gathering
        batch_indices = torch.arange(batch_size, device=input_ids.device)

        # Gather the hidden states of the last token for each layer and batch
        # Shape: (num_layers, batch_size, hidden_dimension)
        last_token_hidden_states = all_hidden_states_tensor[:, batch_indices, last_token_indices, :]

        # Forward the hidden states to the classification layers
        all_output_logits = []
        # Ensure classification_layers has the same length as last_token_hidden_states
        if len(self.classification_layers) != last_token_hidden_states.shape[0]:
             raise ValueError(f"Mismatch in number of layers ({last_token_hidden_states.shape[0]}) and classification heads ({len(self.classification_layers)})")

        for i in range(self.num_layers):
            hidden_state = last_token_hidden_states[i] # (batch_size, hidden_dimension)
            classification_layer = self.classification_layers[i]
            logits = classification_layer(hidden_state) # (batch_size, num_labels)
            all_output_logits.append(logits)

        # Convert to tensor
        all_output_logits_tensor = torch.stack(all_output_logits, dim=0) # (num_layers, batch_size, num_labels)
        return all_output_logits_tensor

def compute_loss(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_layers: int, num_labels: int):
    """
    Compute cross-entropy loss for each layer and the mean loss.

    Args:
        all_layer_logits (torch.Tensor): Shape (num_layers, batch_size, num_labels).
                                         The predicted unnormalized logits from all classification layers.
        labels (torch.Tensor): Shape (batch_size,). Ground truth class labels.
        num_layers (int): Number of layers (should match all_layer_logits.shape[0]).
        num_labels (int): Number of classes (should match all_layer_logits.shape[2]).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - summed_loss (torch.Tensor): A scalar tensor representing the mean loss across all layers and batches.
            - all_layer_loss (torch.Tensor): Shape (num_layers,). The mean loss for each layer across the batch.
    """
    batch_size = labels.shape[0]
    if all_layer_logits.shape[0] != num_layers or all_layer_logits.shape[1] != batch_size or all_layer_logits.shape[2] != num_labels:
        raise ValueError(f"Logits shape mismatch. Expected ({num_layers}, {batch_size}, {num_labels}), got {all_layer_logits.shape}")

    # Reshape logits and labels for cross_entropy
    # Logits: (num_layers * batch_size, num_labels)
    logits_reshaped = all_layer_logits.view(-1, num_labels)

    # Labels: (num_layers * batch_size,)
    # Need to repeat labels for each layer
    labels_repeated = labels.repeat(num_layers)

    # Calculate loss for each instance across all layers
    # Shape: (num_layers * batch_size,)
    instance_loss = F.cross_entropy(logits_reshaped, labels_repeated.long(), reduction='none')

    # Reshape back to (num_layers, batch_size) to calculate per-layer average loss
    loss_per_layer_instance = instance_loss.view(num_layers, batch_size)

    # Calculate average loss per layer
    # Shape: (num_layers,)
    all_layer_mean_loss = loss_per_layer_instance.mean(dim=1)

    # Calculate the overall mean loss for backpropagation
    # This is the mean of all instance losses, equivalent to loss_per_layer_instance.mean()
    summed_loss = instance_loss.mean()

    return summed_loss, all_layer_mean_loss