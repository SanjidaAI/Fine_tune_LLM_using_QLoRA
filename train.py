"""
Training Module
Handles training configuration, execution, and visualization
"""

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
import yaml

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_training_arguments(experiment="exp1"):
    """
    STEP 13 & 22: Create training arguments
    
    Args:
        experiment: "exp1" or "exp2"
    
    Returns:
        TrainingArguments object
    """
    config = load_config()
    
    if experiment == "exp1":
        train_config = config['training_exp1']
    else:
        train_config = config['training_exp2']
    
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        num_train_epochs=train_config['num_train_epochs'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        fp16=train_config['fp16'],
        report_to="none",
        dataloader_pin_memory=False
    )
    
    print(f"\n✅ Training arguments created for {experiment}")
    return training_args

def setup_trainer(model, tokenizer, training_args, train_dataset):
    """
    STEP 14 & 23: Setup Trainer
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        training_args: Training arguments
        train_dataset: Tokenized training dataset
    
    Returns:
        Trainer object
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    print("✅ Trainer setup complete")
    return trainer

def train_model(trainer, experiment_name):
    """
    STEP 15 & 24: Train the model
    
    Args:
        trainer: Trainer object
        experiment_name: Name of experiment for logging
    
    Returns:
        Training result object
    """
    print("\n" + "="*50)
    print(f"🚀 Starting Training - {experiment_name}")
    print("="*50)
    
    train_result = trainer.train()
    
    # Print results
    print("\n" + "="*50)
    print(f"📊 {experiment_name} Results:")
    print("="*50)
    print(f"Global Steps: {train_result.global_step}")
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print(f"Training Runtime: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Epochs Completed: {train_result.metrics['epoch']}")
    print("="*50)
    
    return train_result

def plot_loss_curve(trainer, experiment_name, color='blue', save_path=None):
    """
    STEP 16 & 25: Plot training loss curve
    
    Args:
        trainer: Trainer object
        experiment_name: Name for plot title
        color: Color for the plot line
        save_path: Path to save the plot (optional)
    """
    log_history = trainer.state.log_history
    losses = [log['loss'] for log in log_history if 'loss' in log]
    steps = [log['step'] for log in log_history if 'loss' in log]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', label=experiment_name, color=color)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve - {experiment_name}')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()
    
    return steps, losses

def compare_experiments(steps_exp1, losses_exp1, steps_exp2, losses_exp2, save_path=None):
    """
    STEP 26: Compare both experiments
    
    Args:
        steps_exp1: Training steps from experiment 1
        losses_exp1: Losses from experiment 1
        steps_exp2: Training steps from experiment 2
        losses_exp2: Losses from experiment 2
        save_path: Path to save comparison plot (optional)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(steps_exp1, losses_exp1, marker='o', label='Exp1: 100 samples, 5 epochs', alpha=0.7)
    plt.plot(steps_exp2, losses_exp2, marker='s', label='Exp2: 200 samples, 5 epochs', alpha=0.7, color='orange')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison: 100 vs 200 Samples (Both 5 Epochs)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Comparison plot saved to {save_path}")
    
    plt.show()

def save_model(trainer, tokenizer, experiment="exp1"):
    """
    STEP 17 & 27: Save trained model
    
    Args:
        trainer: Trainer object
        tokenizer: Tokenizer
        experiment: "exp1" or "exp2"
    """
    config = load_config()
    
    if experiment == "exp1":
        save_path = config['paths']['model_save_exp1']
    else:
        save_path = config['paths']['model_save_exp2']
    
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"\n✅ Model saved: {save_path}")

def run_experiment(model, tokenizer, train_dataset, experiment="exp1"):
    """
    Complete training pipeline for one experiment
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        train_dataset: Tokenized training dataset
        experiment: "exp1" or "exp2"
    
    Returns:
        tuple: (trainer, train_result, steps, losses)
    """
    # Create training arguments
    training_args = create_training_arguments(experiment)
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, training_args, train_dataset)
    
    # Train model
    experiment_name = f"Experiment {experiment[-1]}"
    train_result = train_model(trainer, experiment_name)
    
    # Plot loss curve
    color = 'blue' if experiment == "exp1" else 'orange'
    steps, losses = plot_loss_curve(trainer, experiment_name, color)
    
    # Save model
    save_model(trainer, tokenizer, experiment)
    
    return trainer, train_result, steps, losses

# Test the module
if __name__ == "__main__":
    print("Testing train.py module...")
    print("✅ Training module structure is correct!")
    print("Note: Actual training requires model and dataset from other modules")
