import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import AutoModelForCausalLM

class SlicedGemma(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B", num_labels: int = 2):
        super(SlicedGemma, self).__init__()

        print(f"Loading {model_name} model...")
        self.gemma = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        # freeze gemma model parameters
        for param in self.gemma.parameters():
            param.requires_grad = False
        print("Model loaded and parameters frozen.")

        self.num_layers = len(self.gemma.model.layers)
        print(f"Number of layers: {self.num_layers}")
        self.hidden_dimension = self.gemma.config.hidden_size
        print(f"Hidden dimension: {self.hidden_dimension}")
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])
    
    def forward(self, input_ids, attention_mask: np.ndarray = None):
        outputs = self.gemma(input_ids, attention_mask=attention_mask)
        
        # The model outputs are returned in the form of a tuple. 
        # The first item is the sequence of hidden-states at the output of the last layer.
        # Hidden states from all layers are available in the outputs.hidden_states
        all_hidden_states = outputs.hidden_states  # shape: (num_layers+1, batch_size, sequence_length, hidden_dimension)

        # convert all_hidden_states from tuple to tensor 
        # skip the first element which is the embedding layer
        all_hidden_states = torch.stack([tensor for tensor in all_hidden_states[1:]])

        ############# Use the last token of each sequence for classification #############

        # get the last position of tokens
        if attention_mask is not None:  # attention_mask shape: (batch_size, sequence_length)
            last_token_positions = attention_mask.sum(dim=1) - 1  # (batch_size)
        else:
            batch_size = input_ids.shape[0]
            seq_length = input_ids.shape[1]
            last_token_positions = torch.tensor([seq_length - 1] * batch_size, device=input_ids.device)
        
        # create a range array for the batch dimension
        batch_range = torch.arange(all_hidden_states.shape[1], device=input_ids.device)
        
        # select the last token for every layer and batch
        new_hidden_states = all_hidden_states[:, batch_range, last_token_positions]  # (num_layers, batch_size, hidden_dimension)

        # forward the hidden states to the classification layers
        # classification layer is of shape (hidden_dimension, num_labels)
        # new hidden_states is of shape (num_layers, batch_size, hidden_dimension)
        all_output_logits = []
        for hidden_state, classification_layer in zip(new_hidden_states, self.classification_layers):
            logits = classification_layer(hidden_state)  # (batch_size, num_labels)
            all_output_logits.append(logits)  # (num_layers, batch_size, num_labels)
        
        # convert to tensor
        all_output_logits_tensor = torch.stack(all_output_logits, dim=0)
        return all_output_logits_tensor  # (num_layers, batch_size, num_labels)
    
def compute_loss(all_layer_logits: torch.Tensor, labels: torch.Tensor, num_layers: int, num_labels: int):
    """
    Arg:
        all_layer_logits <num_layers, batch_size, num_labels>: The predicted unnormalized logits of the model for all layers.
        labels <batch_size>: Ground truth class labels
    
    Returns:
        A tuple of summed_loss (1D tensor) and all_layer_loss (num_layers)
    """
    # replicate labels to all layers
    labels = labels.clone().detach().repeat(num_layers)  # (batch_size) -> (num_layers*batch_size)

    ########### compute cross entropy loss ###########
    # collapse all dimensions for calculating loss
    labels = labels.reshape(-1)  # (num_layers*batch_size) -> 1D tensor
    
    # collapse all dimensions except the last one for calculating loss
    logits = all_layer_logits.reshape(-1, num_labels)  # (num_layers, batch_size, num_labels) -> (num_layers*batch_size, num_labels)
    
    # calculate loss
    all_layer_loss = F.cross_entropy(logits, labels.long(), reduction='none')  # (num_layers*batch_size)
    
    # reshape and mean over batch dimension
    all_layer_loss = all_layer_loss.reshape(num_layers, -1).mean(dim=1)  # (num_layers)
    
    # mean loss for backpropagation
    # e.g. loss = (u + v) / 2 -> loss with respective to the weights are u'/2 + v'/2
    summed_loss = all_layer_loss.mean()  # (num_layers) -> (1D tensor)

    return summed_loss, all_layer_loss  # (1D tensor), # (num_layers)