Fine-Tuning LLM using QLoRA - Assignment Report

Student: SanjidaAI
Date: January 21, 2026
GitHub: https://github.com/SanjidaAI/Fine_tune_LLM_using_QLoRA
Colab: https://colab.research.google.com/drive/1ieE7tlkycWlka6tHewpuP18m4RjEVJrJ


1. Which Topic I Chose and Why
I chose QLoRA (Quantized Low-Rank Adaptation) for fine-tuning Large Language Models because it makes advanced AI training accessible on limited hardware.
Why QLoRA?

4-bit Quantization: Reduces memory by 75% without quality loss
LoRA: Trains only 0.3% of parameters, dramatically reducing compute needs
Accessibility: Enables fine-tuning billion-parameter models on free Google Colab GPUs

Traditional fine-tuning requires 40GB+ GPU memory. QLoRA lets me do it on free 15GB Colab GPU. This democratizes AI - what needed expensive cloud infrastructure is now free.

2. Dataset Description
Dataset: OpenAssistant Guanaco (timdettmers/openassistant-guanaco)
Characteristics:

Total: ~10,000 conversational samples
Format: Question-answer instruction pairs
Quality: Human-curated for accuracy
Content: Programming, science, general knowledge

Why this dataset?

Designed for instruction fine-tuning
High-quality responses
Diverse topics for general chatbots
Well-structured for QLoRA training

My Experiments:

Experiment 1: 100 samples, 5 epochs
Experiment 2: 200 samples, 5 epochs

This allowed me to measure the impact of dataset size on performance.

3. How I Implemented It
Base Model
TinyLlama-1.1B-Chat-v1.0

Small enough for quick training (1.1B parameters)
Already instruction-tuned
Fully open-source

QLoRA Configuration
Quantization:
- Type: 4-bit NF4
- Compute dtype: float16
- Double quantization: Yes
- Memory: 4.4GB → 1.1GB (75% reduction)

LoRA Parameters:
- Rank (r): 16
- Alpha: 32
- Target modules: q_proj, v_proj
- Dropout: 0.05
- Trainable params: 4,915,200 (0.3%)
Training Setup
- Batch size: 1 (gradient accumulation: 4)
- Learning rate: 2e-4
- Epochs: 5
- Max length: 256 tokens
- Precision: FP16
- Optimizer: AdamW
Implementation Steps (28 Total)

Install libraries (datasets, transformers, peft, bitsandbytes)
Load OpenAssistant dataset
Setup 4-bit quantization
Load TinyLlama base model
Attach LoRA adapters
Tokenize data (max_length=256)
Train Experiment 1 (100 samples)
Evaluate and visualize results
Reload fresh model
Train Experiment 2 (200 samples)
Compare results

Key Design: Reloaded fresh base model for Experiment 2 to ensure fair comparison.

4. Results
Quantitative Results
MetricExperiment 1Experiment 2ImprovementDataset100 samples200 samples2xSteps1252502xFinal Loss1.67161.5787-5.5% Training Time~20  min~30 min Proportional
Key Finding: Doubling training data = 5.5% loss reduction
Loss Curves

Both experiments: Smooth convergence, no overfitting
Experiment 2: Consistently lower loss throughout training
Clear visual difference in comparison graph

Qualitative Results
Test 1: "Explain what is a variable in programming."

Exp 1: Basic definition, limited detail
Exp 2: Comprehensive, structured, with examples

Test 2: "How do computers store data?"

Exp 1: Simple surface-level answer
Exp 2: Detailed explanation, better coherence

Conclusion: Same 2 questions applied to both experiments. Answer patterns are different. Experiment 2 shows better performance and provides more structured answers for fine-tuning with more data (200 samples).

5. Problems Faced and How I Solved Them
Problem 1: GPU Memory Limit
Issue: Colab free tier = 15GB, traditional fine-tuning needs 20-30GB
Solution:

4-bit quantization (75% memory reduction)
Batch size = 1 with gradient accumulation
FP16 precision
Result: Fit in 15GB successfully ✓

Problem 2: Missing Padding Token
Issue: AttributeError: tokenizer has no pad_token
Solution:
pythontokenizer.pad_token = tokenizer.eos_token
Result: Batch processing works perfectly ✓
Problem 3: Training Time
Issue: Colab disconnects after 12 hours
Solution:

Used small datasets (100-200 samples)
Checkpointing every 50 steps
Each experiment < 30 minutes
Result: Completed within time limits ✓

Problem 4: Fair Comparison
Issue: How to compare experiments fairly?
Solution:

Reloaded fresh base model for Exp 2
Same hyperparameters
Only changed: dataset size
Result: Valid scientific comparison ✓

Problem 5: Visualization
Issue: Needed graphs in GitHub repo
Solution:
pythonplt.savefig('loss_curve.png', dpi=300)
Result: Beautiful graphs saved and uploaded ✓

6. Conclusions
Achievements
 Fine-tuned 1.1B model on free Colab GPU
 5.5% improvement with 2x data
 Proved QLoRA works for resource-constrained training
 Complete reproducible implementation
Key Learnings

Data matters: Even 100→200 samples = noticeable improvement
QLoRA works: 4-bit + LoRA = accessible AI for everyone
Proper experiment design: Fresh model reload ensures valid results
Documentation: Clear code + visualizations = professional project

Limitations

Small dataset (100-200 samples)
Could test 500-1000+ samples
Could try other base models
Could add automated metrics (BLEU, ROUGE)

Future Work

Scale to larger datasets
Hyperparameter tuning
Try different models (Llama 2, Mistral)
Deploy as real chatbot


7. References

Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs"
Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
OpenAssistant Guanaco: https://huggingface.co/datasets/timdettmers/openassistant-guanaco
TinyLlama: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0


End of Report
