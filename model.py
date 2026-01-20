"""
Model Loading and Configuration Module
Handles model loading, quantization, and LoRA setup
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import yaml

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_gpu():
    """
    STEP 4: Check GPU availability
    """
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")
    return torch.cuda.is_available()

def create_bnb_config():
    """
    STEP 5: Create QLoRA 4-bit quantization configuration
    
    Returns:
        BitsAndBytesConfig object
    """
    config = load_config()
    quant_config = config['quantization']
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
    )
    
    print("\n✅ QLoRA 4-bit config created")
    return bnb_config

def load_base_model(bnb_config):
    """
    STEP 6: Load base model (TinyLlama) with quantization
    
    Args:
        bnb_config: BitsAndBytesConfig object
    
    Returns:
        Base model
    """
    config = load_config()
    model_name = config['model']['name']
    
    print(f"\nLoading base model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("✅ Base model loaded successfully")
    return model

def load_tokenizer():
    """
    STEP 7: Load tokenizer
    
    Returns:
        Tokenizer object
    """
    config = load_config()
    model_name = config['model']['name']
    
    print(f"\nLoading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Tokenizer loaded successfully")
    return tokenizer

def create_lora_config():
    """
    STEP 8: Setup LoRA configuration
    
    Returns:
        LoraConfig object
    """
    config = load_config()
    lora_params = config['lora']
    
    lora_config = LoraConfig(
        r=lora_params['r'],
        lora_alpha=lora_params['lora_alpha'],
        target_modules=lora_params['target_modules'],
        lora_dropout=lora_params['lora_dropout'],
        bias=lora_params['bias'],
        task_type=lora_params['task_type']
    )
    
    print("\n✅ LoRA configuration created")
    return lora_config

def attach_lora_to_model(model, lora_config):
    """
    STEP 9: Attach LoRA adapters to the model
    
    Args:
        model: Base model
        lora_config: LoRA configuration
    
    Returns:
        Model with LoRA adapters attached
    """
    print("\nAttaching LoRA adapters to model...")
    
    model = get_peft_model(model, lora_config)
    
    print("\n✅ LoRA attached successfully")
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()
    
    return model

def setup_model_and_tokenizer():
    """
    Complete model setup pipeline
    Combines STEPS 4-9
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("\n" + "="*50)
    print("MODEL SETUP PIPELINE")
    print("="*50)
    
    # Check GPU
    check_gpu()
    
    # Create configs
    bnb_config = create_bnb_config()
    lora_config = create_lora_config()
    
    # Load model and tokenizer
    model = load_base_model(bnb_config)
    tokenizer = load_tokenizer()
    
    # Attach LoRA
    model = attach_lora_to_model(model, lora_config)
    
    print("\n" + "="*50)
    print("✅ MODEL SETUP COMPLETE")
    print("="*50)
    
    return model, tokenizer

def reload_fresh_model():
    """
    Reload a fresh base model for new experiment
    Used in STEP 19 for Experiment 2
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("\n🔄 Reloading fresh base model for new experiment...")
    return setup_model_and_tokenizer()

# Test the module
if __name__ == "__main__":
    print("Testing model.py module...")
    model, tokenizer = setup_model_and_tokenizer()
    print("\n✅ Model module working correctly!")
