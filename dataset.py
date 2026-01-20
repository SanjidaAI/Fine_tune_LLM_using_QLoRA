"""
Dataset Loading and Preprocessing Module
Handles dataset loading, tokenization, and sample selection
"""

from datasets import load_dataset
import yaml

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_and_inspect_dataset():
    """
    STEP 2 & 3: Load and inspect the dataset
    Returns the full dataset
    """
    config = load_config()
    dataset_name = config['dataset']['name']
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    print(f"\nFull dataset size: {len(dataset['train'])}")
    print("\nSample data structure:")
    print(dataset["train"][0])
    
    return dataset

def tokenize_function(examples, tokenizer, max_length=256):
    """
    STEP 10: Tokenization function
    Tokenizes text with truncation and padding
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

def prepare_dataset(dataset, tokenizer, num_samples, experiment_name="exp1"):
    """
    STEP 11 & 20: Prepare dataset with specific number of samples
    
    Args:
        dataset: Full dataset
        tokenizer: Tokenizer object
        num_samples: Number of samples to select (100 or 200)
        experiment_name: Name of experiment ("exp1" or "exp2")
    
    Returns:
        Tokenized dataset ready for training
    """
    print(f"\n{'='*50}")
    print(f"Preparing dataset for {experiment_name}")
    print(f"{'='*50}")
    
    # Select subset of samples
    small_dataset = dataset["train"].select(range(num_samples))
    print(f"Training samples: {len(small_dataset)}")
    
    # Tokenize dataset
    tokenized_dataset = small_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    print(f"✅ Dataset prepared with {num_samples} samples")
    return tokenized_dataset

def get_experiment_datasets(dataset, tokenizer):
    """
    Prepare both experiment datasets (100 and 200 samples)
    
    Returns:
        tuple: (tokenized_dataset_100, tokenized_dataset_200)
    """
    config = load_config()
    
    # Experiment 1: 100 samples
    exp1_samples = config['dataset']['exp1_samples']
    tokenized_dataset_100 = prepare_dataset(
        dataset, tokenizer, exp1_samples, "Experiment 1"
    )
    
    # Experiment 2: 200 samples
    exp2_samples = config['dataset']['exp2_samples']
    tokenized_dataset_200 = prepare_dataset(
        dataset, tokenizer, exp2_samples, "Experiment 2"
    )
    
    return tokenized_dataset_100, tokenized_dataset_200

# Test the module
if __name__ == "__main__":
    print("Testing dataset.py module...")
    dataset = load_and_inspect_dataset()
    print("\n✅ Dataset module working correctly!")
