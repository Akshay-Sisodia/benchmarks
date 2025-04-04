# Benchmark Suite for Language Models

This repository contains a comprehensive suite of benchmarks for evaluating large language models (LLMs) across various tasks and capabilities.

## 📋 Overview

This benchmark suite includes multiple evaluation tasks designed to test different aspects of language model performance:

- **TruthfulQA**: Evaluates model's ability to provide truthful answers
- **MMLU** (Massive Multitask Language Understanding): Tests model knowledge across various academic subjects
- **GSM8K** (Grade School Math 8K): Assesses mathematical reasoning capabilities
- **Coin Flip**: Tests model's understanding of probability and randomness
- **BigBench**: Collection of diverse language tasks from the Beyond the Imitation Game benchmark

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)

### Installation

1. Clone the repository:
```bash
git clone https:/github.com/Akshay-Sisodia/benchmarks/
cd benchmarks
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

Key dependencies include:
- vllm (≥0.2.0): High-performance LLM inference
- PyTorch (≥2.0.0): Deep learning framework
- Transformers: Hugging Face transformers library
- NLTK (≥3.8.1): Natural Language Processing toolkit
- Other utilities: pandas, numpy, tqdm, etc.

## 🔧 Usage

The benchmarks can be run using their respective Python scripts. Here's a detailed example using the TruthfulQA benchmark:

```bash
python benchmark_truthfulqa.py \
    --model-id "meta-llama/Llama-2-7b-chat-hf" \
    --batch-size 4 \
    --output-dir "results" \
    --system-prompt "You are a helpful AI assistant that answers questions truthfully and accurately." \
    --max-tokens 100 \
    --temperature 0.2 \
    --top-p 0.9 \
    --top-k 50 \
    --seed 42 \
    --log-dir "logs" \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1
```

### Command Line Arguments

- `--model-id`: HuggingFace model ID or local path (default: "meta-llama/Llama-2-7b-chat-hf")
- `--batch-size`: Number of examples to process in parallel (default: 1)
- `--output-dir`: Directory to save benchmark results (default: "results")
- `--system-prompt`: System prompt for the model (default shown above)
- `--max-tokens`: Maximum number of tokens in model response (default: 100)
- `--temperature`: Sampling temperature (default: 0.2)
- `--top-p`: Nucleus sampling probability threshold (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)
- `--log-dir`: Directory for logging output (default: "logs")
- `--gpu-memory-utilization`: Target GPU memory utilization (default: 0.9)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 1)
- `--cpu-offload-gb`: Amount of GPU memory to offload to CPU in GB (optional)
- `--dtype`: Model data type (optional, auto-detected if not specified)
- `--enforce-eager`: Force eager execution mode (optional)

The script will:
1. Load the specified model
2. Run the TruthfulQA benchmark
3. Generate a detailed report with metrics including:
   - MC1 (single-answer) accuracy
   - MC2 (multiple-answer) accuracy
   - Generation metrics (ROUGE scores)
   - Runtime performance statistics

Results will be saved in JSON format in the specified output directory.

## 📊 Benchmark Descriptions

### TruthfulQA
- Tests model's ability to provide truthful answers
- Evaluates resistance to common misconceptions
- File: `benchmark_truthfulqa.py`

### MMLU (Massive Multitask Language Understanding)
- Comprehensive evaluation across academic subjects
- Tests knowledge in fields like science, humanities, math, etc.
- File: `benchmark_mmlu.py`

### GSM8K (Grade School Math 8K)
- Focuses on mathematical reasoning and problem-solving
- Tests step-by-step mathematical thinking
- File: `benchmark_gsm8k.py`

### Coin Flip
- Evaluates probabilistic reasoning
- Tests understanding of randomness and basic probability
- File: `benchmark_coin_flip.py`

### BigBench
- Collection of diverse language tasks
- Tests various cognitive and linguistic capabilities
- File: `benchmark_bigbench.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements and bug fixes.


## 📞 Contact

akshay.sisodia2021@vitstudent.ac.in
patilojas.abhijit2021@vitstudent.ac.in