"""
Evaluation Module
Handles model testing and response generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_trained_model(experiment="exp1"):
    """
    Load a trained model with LoRA adapters
    
    Args:
        experiment: "exp1" or "exp2"
    
    Returns:
        tuple: (model, tokenizer)
    """
    config = load_config()
    model_name = config['model']['name']
    
    if experiment == "exp1":
        model_path = config['paths']['model_save_exp1']
    else:
        model_path = config['paths']['model_save_exp2']
    
    print(f"\nLoading trained model from: {model_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"✅ Model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """
    Generate response for a given prompt
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def test_model(model, tokenizer, experiment_name):
    """
    STEP 18 & 28: Test model with predefined questions
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        experiment_name: Name of experiment for display
    """
    print("\n" + "="*50)
    print(f"🧪 Testing {experiment_name} Model:")
    print("="*50)
    
    # Test 1: What is a variable in programming?
    print("\n📝 Test 1 - What is a variable in programming?")
    print("-"*50)
    prompt1 = "### Human: Explain what is a variable in programming. ### Assistant:"
    response1 = generate_response(model, tokenizer, prompt1)
    print(response1)
    
    print("\n" + "-"*50)
    
    # Test 2: How do computers store data?
    print("\n📝 Test 2 - How do computers store data?")
    print("-"*50)
    prompt2 = "### Human: How do computers store data? ### Assistant:"
    response2 = generate_response(model, tokenizer, prompt2)
    print(response2)
    
    print("\n" + "="*50)
    
    return response1, response2

def test_custom_prompt(model, tokenizer, question):
    """
    Test model with a custom question
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        question: Custom question to ask
    
    Returns:
        Generated response
    """
    prompt = f"### Human: {question} ### Assistant:"
    print(f"\n❓ Question: {question}")
    print("-"*50)
    response = generate_response(model, tokenizer, prompt)
    print(response)
    return response

def compare_models(exp1_model, exp1_tokenizer, exp2_model, exp2_tokenizer, question):
    """
    Compare responses from both experiments
    
    Args:
        exp1_model: Model from experiment 1
        exp1_tokenizer: Tokenizer from experiment 1
        exp2_model: Model from experiment 2
        exp2_tokenizer: Tokenizer from experiment 2
        question: Question to compare
    """
    prompt = f"### Human: {question} ### Assistant:"
    
    print("\n" + "="*50)
    print(f"🔄 COMPARISON: {question}")
    print("="*50)
    
    # Experiment 1 response
    print("\n🔵 Experiment 1 (100 samples):")
    print("-"*50)
    response1 = generate_response(exp1_model, exp1_tokenizer, prompt)
    print(response1)
    
    # Experiment 2 response
    print("\n🟠 Experiment 2 (200 samples):")
    print("-"*50)
    response2 = generate_response(exp2_model, exp2_tokenizer, prompt)
    print(response2)
    
    print("\n" + "="*50)
    
    return response1, response2

def evaluate_experiment(experiment="exp1"):
    """
    Complete evaluation pipeline for one experiment
    
    Args:
        experiment: "exp1" or "exp2"
    """
    experiment_name = f"Experiment {experiment[-1]}"
    
    # Load model
    model, tokenizer = load_trained_model(experiment)
    
    # Test with predefined questions
    test_model(model, tokenizer, experiment_name)
    
    return model, tokenizer

def full_comparison():
    """
    Run full comparison between both experiments
    """
    print("\n" + "="*60)
    print("🔬 FULL MODEL COMPARISON: EXPERIMENT 1 vs EXPERIMENT 2")
    print("="*60)
    
    # Load both models
    print("\n📥 Loading Experiment 1 model...")
    exp1_model, exp1_tokenizer = load_trained_model("exp1")
    
    print("\n📥 Loading Experiment 2 model...")
    exp2_model, exp2_tokenizer = load_trained_model("exp2")
    
    # Compare with test questions
    questions = [
        "Explain what is a variable in programming.",
        "How do computers store data?"
    ]
    
    for question in questions:
        compare_models(
            exp1_model, exp1_tokenizer,
            exp2_model, exp2_tokenizer,
            question
        )

# Test the module
if __name__ == "__main__":
    print("Testing evaluate.py module...")
    print("✅ Evaluation module structure is correct!")
    print("Note: Actual evaluation requires trained models from train.py")
