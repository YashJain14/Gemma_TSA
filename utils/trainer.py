import os
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from transformers import Trainer, TrainingArguments, EvalPrediction, PreTrainedTokenizerBase
import wandb
import numpy as np # Import numpy for argmax

# Check if WANDB env vars are set, provide defaults if not
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "yashjain14-nanyang-technological-university-singapore-org") # Default or your W&B username/team
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "NNDL") # Default project name
if WANDB_ENTITY == "your_entity":
    print("Warning: WANDB_ENTITY environment variable not set. Using default 'your_entity'.")

# Attempt login, handle potential errors
try:
    wandb.login()
    print("W&B login successful.")
except Exception as e:
    print(f"W&B login failed: {e}. Check API key or network connection.")
    # Optionally, disable W&B reporting if login fails
    # os.environ["WANDB_DISABLED"] = "true"

# Default optimizer: AdamW
# Adjust batch sizes based on typical GPU memory (e.g., 16-24GB)
# Effective batch size = 32 (adjust as needed)
PER_DEVICE_BATCH_SIZE = 4 # Smaller default, adjust based on GPU memory
GRAD_ACCUM_STEPS = 8      # Adjust to reach effective batch size

training_args = TrainingArguments(
    output_dir='./results',             # output directory
    num_train_epochs=3,               # total number of training epochs
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,  # batch size per device during training
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE*2, # larger eval batch size is often possible
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,   # number of updates steps to accumulate before performing a backward/update pass
    warmup_steps=50,                  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    logging_dir='./logs',             # directory for storing logs
    logging_steps=10,                 # log metrics every X steps
    evaluation_strategy="steps",      # evaluate validation set during training
    eval_steps=100,                   # evaluate every X steps
    save_strategy="steps",            # save checkpoint during training
    save_steps=100,                   # save checkpoint every X steps
    save_total_limit=3,               # limit the total number of checkpoints saved
    load_best_model_at_end=True,      # load the best model found during training at the end
    metric_for_best_model="accuracy", # metric to determine the best model
    report_to="wandb",                # enable logging to W&B
    fp16=True,                        # enable mixed precision training (if GPU supports it)
    # deepspeed='ds_config.json',     # DeepSpeed config file (ensure it exists) - Handled by deepspeed launch
    push_to_hub=False,                # whether to push model to Hugging Face Hub
    label_names=["labels"],           # Ensure HF knows the label column name
    # Remove deepspeed config path from here, it's passed via CLI launcher
)


def compute_metrics(pred: EvalPrediction):
    """Compute metrics using torchmetrics."""
    labels = pred.label_ids
    preds = pred.predictions

    # Handle tuple output (common in some HF models)
    if isinstance(preds, tuple):
        logits = preds[0]
    else:
        logits = preds

    if logits is None or labels is None:
        print("Warning: predictions or labels are None in compute_metrics.")
        return {}

    # Check shapes
    if logits.shape[0] != labels.shape[0]:
         print(f"Warning: Mismatch in prediction ({logits.shape}) and label ({labels.shape}) counts.")
         # Attempt to truncate labels if it's longer (might happen with dataset issues)
         min_len = min(logits.shape[0], labels.shape[0])
         logits = logits[:min_len]
         labels = labels[:min_len]
         if min_len == 0: return {}


    try:
        num_classes = logits.shape[1]
        preds_class = np.argmax(logits, axis=1)

        # Convert to torch tensors on the correct device
        # Important: Ensure metrics are on the same device as the data implicitly is
        # If running distributed, aggregation happens before this, usually on CPU/rank 0
        # For simplicity, let's compute on CPU after potential numpy conversion
        device = "cpu" # Safer default for metrics calculation after potential numpy conversion
        labels_tensor = torch.tensor(labels, device=device).long() # Ensure long type for labels
        preds_tensor = torch.tensor(logits, device=device) # Keep logits for AUROC
        preds_class_tensor = torch.tensor(preds_class, device=device) # Use argmax predictions for others

        # Initialize metrics
        accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        precision = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device) # Use macro avg
        recall = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)    # Use macro avg
        f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)       # Use macro avg
        auroc = AUROC(task="multiclass", num_classes=num_classes).to(device)


        # Calculate metrics
        # Accuracy, Precision, Recall, F1 use class predictions (argmax)
        # AUROC uses the raw probabilities/logits
        accuracy_score = accuracy(preds_class_tensor, labels_tensor)
        precision_score = precision(preds_class_tensor, labels_tensor)
        recall_score = recall(preds_class_tensor, labels_tensor)
        f1_score = f1(preds_class_tensor, labels_tensor)

        # Ensure logits are float for AUROC
        auroc_score = auroc(preds_tensor.float(), labels_tensor)


        return {
            "accuracy": accuracy_score.item(),
            "precision": precision_score.item(),
            "recall": recall_score.item(),
            "f1": f1_score.item(),
            "auroc": auroc_score.item(),
        }
    except Exception as e:
        print(f"Error during metric calculation: {e}")
        print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}, Num classes inferred: {logits.shape[1] if len(logits.shape) > 1 else 'N/A'}")
        return {} # Return empty dict on error


class CustomTrainer(Trainer):
    def __init__(self, *args,
                 run_name: str = None,
                 trainer_args: TrainingArguments = None,
                 num_train_epochs: int = None, # Allow overriding epochs
                 tokenizer: PreTrainedTokenizerBase = None, # Add tokenizer
                 **kwargs):

        # Set default training arguments if not supplied
        current_args = trainer_args if trainer_args else training_args

        # Override epochs if specified
        if num_train_epochs is not None:
            current_args.num_train_epochs = num_train_epochs

        # Specify the run name for wandb logging
        if run_name:
            current_args.run_name = run_name
            # Set WANDB_RUN_NAME env var as well for safety
            os.environ["WANDB_RUN_NAME"] = run_name

        # Initialize the Trainer
        super().__init__(*args,
                         args=current_args,
                         compute_metrics=compute_metrics,
                         tokenizer=tokenizer, # Pass tokenizer to base Trainer
                         **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to use CrossEntropyLoss.
        Handles models that might return loss directly or require manual calculation.
        """
        if "labels" not in inputs:
             raise ValueError("Input dictionary must contain 'labels' for loss computation.")

        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)

        # Some models compute the loss internally when labels are provided
        loss = outputs.get("loss")

        if loss is None:
            # If loss not computed by model, calculate it manually
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("Model output must contain either 'loss' or 'logits'.")

            # Ensure logits and labels are compatible
            if logits.shape[0] != labels.shape[0]:
                 raise ValueError(f"Logits batch size ({logits.shape[0]}) does not match labels batch size ({labels.shape[0]}).")

            loss_fct = torch.nn.CrossEntropyLoss()
            # Reshape logits for CrossEntropyLoss: (batch_size * seq_len, num_labels) if needed
            # Standard SequenceClassification models output (batch_size, num_labels)
            # Ensure labels are Long type
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1).long())

        return (loss, outputs) if return_outputs else loss