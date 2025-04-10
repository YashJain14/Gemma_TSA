import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import LlamaModel, AutoConfig, BitsAndBytesConfig

class SlicedLlama3_2(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", num_labels: int = 2):
        super(SlicedLlama3_2, self).__init__()

        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        
        # Check if model can benefit from 4-bit quantization
        if "llama" in model_name.lower():
            print("Loading Llama model with optimizations for memory efficiency...")
            compute_dtype = getattr(torch, "float16")
            
            # We'll handle external quantization config, but this ensures we set the right dtype
            self.llama = LlamaModel.from_pretrained(
                model_name, 
                config=self.config, 
                torch_dtype=compute_dtype,
            )
        else:
            self.llama = LlamaModel.from_pretrained(model_name, config=self.config)

        # freeze llama model parameters
        for param in self.llama.parameters():
            param.requires_grad = False

        self.num_layers = self.config.num_hidden_layers
        self.hidden_dimension = self.config.hidden_size
        self.num_labels = num_labels
        self.classification_layers = nn.ModuleList([nn.Linear(self.hidden_dimension, self.num_labels) for _ in range(self.num_layers)])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.llama(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # hidden_states tuple: (embedding_output, layer_1_output, ..., layer_n_output)
        all_hidden_states = outputs.hidden_states[1:] # Exclude embedding layer

        # Convert all_hidden_states from tuple to tensor
        all_hidden_states_tensor = torch.stack(all_hidden_states, dim=0) # (num_layers, batch_size, sequence_length, hidden_dimension)

        ############# Use the last token of each sequence for classification #############
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            sequence_lengths = torch.sum(attention_mask, dim=1) - 1
            last_token_indices = sequence_lengths.long()
        else:
            last_token_indices = torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=input_ids.device)

        batch_indices = torch.arange(batch_size, device=input_ids.device)

        # Shape: (num_layers, batch_size, hidden_dimension)
        last_token_hidden_states = all_hidden_states_tensor[:, batch_indices, last_token_indices, :]

        all_output_logits = []
        if len(self.classification_layers) != last_token_hidden_states.shape[0]:
             raise ValueError(f"Mismatch in number of layers ({last_token_hidden_states.shape[0]}) and classification heads ({len(self.classification_layers)})")

        for i in range(self.num_layers):
            hidden_state = last_token_hidden_states[i]
            classification_layer = self.classification_layers[i]
            logits = classification_layer(hidden_state)
            all_output_logits.append(logits)

        all_output_logits_tensor = torch.stack(all_output_logits, dim=0)
        return all_output_logits_tensor

# Re-use the same compute_loss function as SlicedGemma
from .SlicedGemma import compute_loss