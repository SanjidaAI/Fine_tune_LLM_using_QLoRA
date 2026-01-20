# Assignment Topic - Fine-Tuning TinyLlama using QLoRA

##  Project Overview

This project demonstrates **fine-tuning TinyLlama-1.1B** language model using **QLoRA (Quantized Low-Rank Adaptation)** technique. The project compares two experiments with different dataset sizes to analyze the impact of training data volume on model performance.

###  Key Objectives
- Implement QLoRA for efficient fine-tuning with 4-bit quantization
- Compare model performance with **100 samples vs 200 samples**
- Analyze training loss curves and model responses
- Create modular, reusable code structure

---

##  Experiments

### **Experiment 1: 100 Samples, 5 Epochs**
- Training samples: 100
- Number of epochs: 5
- Final training loss: **1.6716**

### **Experiment 2: 200 Samples, 5 Epochs**
- Training samples: 200
- Number of epochs: 5
- Final training loss: **1.5787**

###  Key Finding
Doubling the training data (100 → 200 samples) resulted in **lower training loss** and **better model performance**, demonstrating the importance of sufficient training data for effective fine-tuning.

---

##  Project Structure
```
tinyllama-qlora-finetuning/
│
├── README.md                              # Project documentation (you are here!)
├── requirements.txt                       # Python dependencies
├── config.yaml                            # Configuration parameters
│
├── dataset.py                             # Dataset loading & preprocessing
├── model.py                               # Model setup & LoRA configuration
├── train.py                               # Training pipeline & visualization
├── evaluate.py                            # Model evaluation & testing
│
├── Assignment_Original_Colab.ipynb        # Original Colab notebook
└── REPORT.pdf                             # Detailed project report (1-2 pages)
```

---



### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SanjidaAI/tinyllama-qlora-finetuning.git
cd tinyllama-qlora-finetuning
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Training (Optional - runs both experiments)
```python
# Full pipeline example
from dataset import load_and_inspect_dataset, get_experiment_datasets
from model import setup_model_and_tokenizer, reload_fresh_model
from train import run_experiment, compare_experiments
from evaluate import evaluate_experiment

# Load dataset
dataset = load_and_inspect_dataset()

# Setup model
model, tokenizer = setup_model_and_tokenizer()

# Prepare datasets
tokenized_dataset_100, tokenized_dataset_200 = get_experiment_datasets(dataset, tokenizer)

# Run Experiment 1
trainer1, result1, steps1, losses1 = run_experiment(model, tokenizer, tokenized_dataset_100, "exp1")

# Reload fresh model for Experiment 2
model, tokenizer = reload_fresh_model()

# Run Experiment 2
trainer2, result2, steps2, losses2 = run_experiment(model, tokenizer, tokenized_dataset_200, "exp2")

# Compare experiments
compare_experiments(steps1, losses1, steps2, losses2)

# Evaluate both models
evaluate_experiment("exp1")
evaluate_experiment("exp2")
```

### 4️⃣ Evaluate Trained Models
```python
from evaluate import full_comparison

# Compare both experiments
full_comparison()
```

---

##  Configuration

All hyperparameters are stored in `config.yaml`:

- **Model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **LoRA rank (r):** 16
- **LoRA alpha:** 32
- **Learning rate:** 2e-4
- **Batch size:** 1 (with gradient accumulation steps: 4)
- **Epochs:** 5
- **Quantization:** 4-bit (NF4)

---

##  Results Summary

| Experiment | Samples | Epochs | Training Loss | Runtime |
|------------|---------|--------|---------------|---------|
| Exp 1      | 100     | 5      | 1.6716        | ~X min  |
| Exp 2      | 200     | 5      | 1.5787        | ~Y min  |

**Conclusion:** Experiment 2 (200 samples) achieved **better performance** with lower training loss, highlighting the importance of adequate training data.

---

##  Technologies Used

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Model loading and training
- **PEFT** - Parameter-Efficient Fine-Tuning (LoRA)
- **BitsAndBytes** - 4-bit quantization
- **Datasets** (Hugging Face) - Dataset management
- **Matplotlib** - Visualization

---

##  Dataset

**Dataset:** [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

This dataset contains high-quality conversational data suitable for instruction fine-tuning.

---

##  Testing

The models are tested with two questions:

1. **"Explain what is a variable in programming."**
2. **"How do computers store data?"**

Results show that Experiment 2 (200 samples) provides more structured and coherent responses.

---

##  Files Description

| File | Purpose |
|------|---------|
| `dataset.py` | Loads dataset, handles tokenization, prepares train samples |
| `model.py` | Configures QLoRA, loads TinyLlama, attaches LoRA adapters |
| `train.py` | Sets up training arguments, runs training, plots loss curves |
| `evaluate.py` | Loads trained models, generates responses, compares outputs |
| `config.yaml` | Stores all hyperparameters and configurations |
| `requirements.txt` | Lists all Python package dependencies |

---

##  Author

**Sanjida Haque**  
**Course:** [Deep Learning & Generative AI]  
**Assignment:** Fine-Tuning LLM using QLoRA  
**Date:** 21st January 2026

---

##  License

This project is for educational purposes.

---

##  Acknowledgments

- **Hugging Face** for Transformers and PEFT libraries
- **TinyLlama Team** for the base model
- **Tim Dettmers** for QLoRA technique and dataset
