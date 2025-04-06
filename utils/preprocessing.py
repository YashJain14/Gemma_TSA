from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict

def train_val_test_split(dataset: Dataset | DatasetDict, seed: int = 42):
    """
    Splits the dataset into training, validation, and test sets.
    Handles cases where 'validation' or 'test' might be missing.
    """
    if "test" not in dataset:
        # If no test set, split train into train/test
        train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=seed)
        dataset["train"] = train_test_split["train"]
        dataset["test"] = train_test_split["test"]

    if "validation" not in dataset:
        # If no validation set, split (new) train into train/validation
        train_val_split = dataset["train"].train_test_split(test_size=0.2, seed=seed)
        dataset["train"] = train_val_split["train"]
        dataset["validation"] = train_val_split["test"]

    return (dataset["train"], dataset["validation"], dataset["test"])


def tokenize(dataset: Dataset | DatasetDict,
             tokenizer: PreTrainedTokenizerBase, # Pass tokenizer object
             input_col_name: str = "text",
             max_length: int = 512):
    """Tokenizes the dataset using the provided tokenizer."""

    print(f"Tokenizing column '{input_col_name}' with tokenizer '{tokenizer.name_or_path}' (padding_side='{tokenizer.padding_side}')")

    def _tokenize(examples):
        # Tokenizer handles padding and truncation
        return tokenizer(
            examples[input_col_name],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    # Ensure necessary columns exist before mapping
    cols_to_remove = [col for col in dataset['train'].column_names if col not in ["input_ids", "attention_mask", "label", input_col_name]]

    tokenized_datasets = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=cols_to_remove # Remove original text and other unused cols
    ).with_format("torch") # Ensure output is torch tensors

    # Verify columns after tokenization
    print("Columns after tokenization:", tokenized_datasets['train'].column_names)
    required_cols = {"input_ids", "attention_mask", "label"}
    if not required_cols.issubset(tokenized_datasets['train'].column_names):
         print(f"Warning: Missing required columns after tokenization. Expected {required_cols}, got {tokenized_datasets['train'].column_names}")
         # Attempt to select required columns anyway, map might behave differently based on dataset source
         # This might fail if columns truly don't exist
         try:
             tokenized_datasets = tokenized_datasets.select_columns(["input_ids", "attention_mask", "label"])
         except ValueError as e:
              print(f"Error selecting columns: {e}")
              print("Please check dataset structure and tokenizer output.")
              raise e


    return tokenized_datasets


def subset_dataset(dataset: Dataset | DatasetDict,
                   size: int,
                   seed: int = 42):
    """Selects a random subset of the dataset."""
    if size >= len(dataset):
        print(f"Subset size ({size}) is >= dataset size ({len(dataset)}). Returning original dataset.")
        return dataset
    shuffled_dataset = dataset.shuffle(seed=seed)
    new_dataset = shuffled_dataset.select(range(size))
    print(f"Subsetted dataset from {len(dataset)} to {len(new_dataset)} samples.")
    return new_dataset