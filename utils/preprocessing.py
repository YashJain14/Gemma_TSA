from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
import torch

def tokenize(dataset, tokenizer_name, input_col_name="text", max_length=512):
    """
    Tokenize the dataset using the specified tokenizer.
    
    Args:
        dataset: The dataset to tokenize
        tokenizer_name: Name or path of the tokenizer
        input_col_name: Column name containing the text to tokenize
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Handle tokenizers without pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    def _tokenize(examples):
        return tokenizer(
            examples[input_col_name], 
            padding='max_length', 
            truncation=True, 
            max_length=max_length
        )
    
    print(f"Tokenizing dataset with column: {input_col_name}")
    tokenized_datasets = dataset.map(
        _tokenize, 
        batched=True, 
        desc="Tokenizing"
    ).select_columns(["input_ids", "attention_mask", "label"]).with_format("torch")
    
    return tokenized_datasets

def train_val_test_split(dataset, val_size=0.1, test_size=0.1, seed=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if isinstance(dataset, DatasetDict):
        # If dataset already has train/test splits
        train_dataset = dataset["train"]
        
        if "validation" in dataset:
            val_dataset = dataset["validation"]
        else:
            # Split train into train and validation
            splits = train_dataset.train_test_split(test_size=val_size, seed=seed)
            train_dataset = splits["train"]
            val_dataset = splits["test"]
        
        test_dataset = dataset["test"] if "test" in dataset else None
        
        # If no test set is provided, split validation further
        if test_dataset is None:
            splits = val_dataset.train_test_split(test_size=0.5, seed=seed)
            val_dataset = splits["train"]
            test_dataset = splits["test"]
    else:
        # If dataset is a single Dataset object
        test_split = dataset.train_test_split(test_size=test_size, seed=seed)
        train_val = test_split["train"]
        test_dataset = test_split["test"]
        
        train_val_split = train_val.train_test_split(
            test_size=val_size/(1-test_size),  # adjusted to account for previous split
            seed=seed
        )
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
    
    print(f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset

def subset_dataset(dataset, size=25000, seed=42):
    """
    Create a subset of the dataset, useful for large datasets like Yelp.
    
    Args:
        dataset: Dataset to subset
        size: Number of examples to include
        seed: Random seed for reproducibility
        
    Returns:
        Subsetted dataset
    """
    if len(dataset) <= size:
        return dataset
    
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.select(range(size))