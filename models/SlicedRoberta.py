import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import RobertaModel, AutoConfig

class SlicedRoberta(nn.Module):
    def __init__(self, model_name: str = "roberta-base", num_labels: int = 2):
        super(SlicedRoberta, self).__init__()

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)

        # freeze roberta model parameters
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Use config values for consistency
        self.num_layers = self.config.num_hidden_layers
        self.hidden_dimension = self.config.hidden_size
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # hidden_states tuple: (embedding_output, layer_1_output, ..., layer_n_output)
        all_hidden_states = outputs.hidden_states[1:] # Exclude embedding layer

        # Convert all_hidden_states from tuple to tensor
        all_hidden_states_tensor = torch.stack(all_hidden_states, dim=0) # (num_layers, batch_size, sequence_length, hidden_dimension)

        ############# Use the first token (CLS) of each sequence for classification #############
        # Shape: (num_layers, batch_size, hidden_dimension)
        first_token_hidden_states = all_hidden_states_tensor[:, :, 0, :]

        # Forward the hidden states to the classification layers
        all_output_logits = []
        if len(self.classification_layers) != first_token_hidden_states.shape[0]:
             raise ValueError(f"Mismatch in number of layers ({first_token_hidden_states.shape[0]}) and classification heads ({len(self.classification_layers)})")

        for i in range(self.num_layers):
            hidden_state = first_token_hidden_states[i] # (batch_size, hidden_dimension)
            classification_layer = self.classification_layers[i]
            logits = classification_layer(hidden_state) # (batch_size, num_labels)
            all_output_logits.append(logits)

        # Convert to tensor
        all_output_logits_tensor = torch.stack(all_output_logits, dim=0) # (num_layers, batch_size, num_labels)
        return all_output_logits_tensor

# Re-use the same compute_loss function as SlicedGemma
from .SlicedGemma import compute_loss