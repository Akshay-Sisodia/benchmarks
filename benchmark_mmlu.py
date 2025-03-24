#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import requests
import numpy as np
import torch
import re
import subprocess
import io
import sys
import zipfile
from pathlib import Path
import datasets

# Suppress warnings and configure environment variables upfront
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"  # Disable vLLM's progress bar
os.environ["VLLM_DISABLE_TQDM"] = "1"  # Disable vLLM's tqdm
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', message='.*GetPrototype.*')

# Global lock for thread-safety
thread_lock = threading.RLock()

# MMLU Categories
MMLU_CATEGORIES = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous",
    "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions"
]

# Answer choices mapping
ANSWER_MAPPING = {0: "A", 1: "B", 2: "C", 3: "D"}
REVERSE_ANSWER_MAPPING = {"A": 0, "B": 1, "C": 2, "D": 3}

# Set up logging
def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """Setup detailed logging for the benchmark run."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"benchmark_{run_id}.log")
    
    logger = logging.getLogger("vllm_benchmark")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@lru_cache(maxsize=1)
def load_mmlu(category: str, split: str = "test") -> 'datasets.Dataset':
    """Load a specific category of the MMLU dataset with caching."""
    try:
        # Configure dataset loading for speed
        datasets.config.HF_DATASETS_OFFLINE = 0
        datasets.config.IN_MEMORY_MAX_SIZE = 0  # Disable size limit for in-memory datasets
        datasets.config.DOWNLOADED_DATASETS_PATH = os.path.expanduser("~/.cache/mmlu")
        
        # Load the dataset with minimal verification and caching
        dataset = datasets.load_dataset(
            "cais/mmlu",
            name=category,
            split=split,
            verification_mode="no_checks",
            trust_remote_code=True,
            cache_dir=os.path.expanduser("~/.cache/mmlu")
        )
        
        return dataset
    except Exception as e:
        error_msg = f"Failed to load MMLU category '{category}': {str(e)}"
        print(error_msg)
        raise ValueError(error_msg)

def load_all_mmlu_categories(categories: List[str], split: str = "test") -> Dict[str, 'datasets.Dataset']:
    """Load multiple MMLU categories in parallel using ThreadPoolExecutor."""
    datasets_dict = {}
    num_workers = min(16, len(categories))
    
    def load_category(category):
        try:
            dataset = load_mmlu(category, split)
            if dataset is not None:
                return category, dataset
        except Exception as e:
            print(f"Failed to load category {category}: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(load_category, categories))
    
    for result in results:
        if result:
            category, dataset = result
            datasets_dict[category] = dataset
    
    return datasets_dict

# Extract answer from model response for MMLU
def extract_answer(response: str) -> Optional[str]:
    """Extract the letter answer (A, B, C, or D) from the model's response."""
    if not response or not isinstance(response, str):
        return None
    
    response = response.strip().upper()
    
    # First try to find standalone A, B, C, or D
    pattern = r'\b[ABCD]\b'
    matches = re.findall(pattern, response)
    if matches:
        return matches[0]
    
    # Try to find "Answer: X" pattern
    pattern = r'(?:ANSWER|ANS|CHOICE)(?:\s*(?:IS|:))?\s*([ABCD])\b'
    match = re.search(pattern, response)
    if match:
        return match.group(1)
    
    return None

# Format few-shot examples for MMLU
def format_mmlu_question(question: str, choices: List[str], include_answer: bool = False, answer: Optional[int] = None) -> str:
    """Format an MMLU question with its choices."""
    formatted_question = f"Question: {question}\n\n"
    for idx, choice in enumerate(choices):
        formatted_question += f"{ANSWER_MAPPING[idx]}. {choice}\n"
    if include_answer and answer is not None:
        formatted_question += f"\nThe correct answer is: {ANSWER_MAPPING[answer]}"
    return formatted_question

def format_few_shot_examples(examples: List[Dict], system_prompt: str, model: 'LLM') -> str:
    """Format examples as a conversation using the model's chat template."""
    conversation = [{"role": "system", "content": system_prompt}]
    
    for example in examples:
        question = format_mmlu_question(
            example["question"],
            example["choices"],
            include_answer=False
        )
        answer = f"The answer is {ANSWER_MAPPING[example['answer']]}."
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": answer})
    
    try:
        return model.get_tokenizer().apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except (AttributeError, ValueError):
        formatted_prompt = system_prompt + "\n\n"
        for i in range(0, len(conversation)-1, 2):
            if i+1 < len(conversation):
                user_msg = conversation[i+1]["content"]
                assistant_msg = conversation[i+2]["content"] if i+2 < len(conversation) else ""
                formatted_prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        return formatted_prompt

# GPU model to compute capability mapping
GPU_COMPUTE_MAP = {
    "V100": 7.0, "P100": 6.0, "P40": 6.1, "P4": 6.1, "T4": 7.5,
    "A100": 8.0, "A40": 8.6, "A30": 8.0, "A10": 8.6, "A10G": 8.6,
    "A6000": 8.6, "RTX 3090": 8.6, "RTX 3080": 8.6, "RTX 3070": 8.6,
    "RTX 3060": 8.6, "RTX 2080": 7.5, "RTX 2070": 7.5, "RTX 2060": 7.5,
    "GTX 1080": 6.1, "GTX 1070": 6.1, "GTX 1060": 6.1, "H100": 9.0,
    "L40": 8.9, "L4": 8.9, "RTX 4090": 8.9, "RTX 4080": 8.9,
    "RTX 4070": 8.9, "RTX 4060": 8.9
}

@lru_cache(maxsize=1)
def check_gpu_capabilities() -> Dict[str, Any]:
    """Check GPU capabilities including BFloat16 support and compute capability."""
    capabilities = {
        "has_cuda": torch.cuda.is_available(),
        "has_bf16": False,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_names": [],
        "compute_capabilities": [],
        "should_use_eager_attention": False
    }
    
    if not capabilities["has_cuda"]:
        return capabilities
    
    for i in range(capabilities["num_gpus"]):
        device_name = torch.cuda.get_device_name(i)
        capabilities["device_names"].append(device_name)
        
        compute_capability = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=2
            )
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if i < len(lines):
                    compute_capability = float(lines[i].strip())
        except:
            pass
        
        if compute_capability is None:
            for gpu_name, cc in GPU_COMPUTE_MAP.items():
                if gpu_name in device_name:
                    compute_capability = cc
                    break
            if compute_capability is None:
                compute_capability = 7.0
        
        capabilities["compute_capabilities"].append(compute_capability)
        
        if compute_capability >= 8.0:
            capabilities["has_bf16"] = True
        
        if not capabilities["has_bf16"]:
            try:
                with torch.cuda.device(i):
                    test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=f"cuda:{i}")
                    del test_tensor
                    capabilities["has_bf16"] = True
            except:
                pass
    
    capabilities["should_use_eager_attention"] = min(capabilities["compute_capabilities"]) < 8.0
    
    return capabilities

def process_batch(model, batch_data, sampling_params, few_shot_examples, system_prompt):
    """Process a batch of examples efficiently."""
    prompts = []
    for category, example in batch_data:
        question = format_mmlu_question(example["question"], example["choices"])
        if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "apply_chat_template"):
            conversation = [
                {"role": "system", "content": system_prompt},
                *[{"role": "user", "content": format_mmlu_question(ex["question"], ex["choices"]),
                   "role": "assistant", "content": f"The answer is {ANSWER_MAPPING[ex['answer']]}."} 
                  for ex in few_shot_examples],
                {"role": "user", "content": question}
            ]
            prompt = model.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        else:
            prompt = format_few_shot_examples(few_shot_examples, system_prompt, model) + f"User: {question}\nAssistant:"
        prompts.append(prompt)
    
    batch_start = time.time()
    outputs = model.generate(prompts, sampling_params)
    batch_time = time.time() - batch_start
    
    return outputs, batch_time

def optimize_memory_settings(gpu_capabilities, cpu_offload_gb):
    """Optimize memory settings based on GPU capabilities."""
    settings = {
        "dtype": "bfloat16" if gpu_capabilities["has_bf16"] else "float16",
        "gpu_memory_utilization": 1.0 if cpu_offload_gb > 0 else 0.9,
        "enforce_eager": gpu_capabilities["should_use_eager_attention"],
        "prefill_chunk_size": 2048 if gpu_capabilities["compute_capabilities"][0] >= 8.0 else None
    }
    return settings

def calculate_metrics(results, total_correct, total_processed, latencies):
    """Calculate all metrics properly with error handling."""
    try:
        # Overall metrics
        results["metrics"]["overall"].update({
            "total_correct": total_correct,
            "total_examples": total_processed,
            "accuracy": total_correct / total_processed if total_processed > 0 else 0.0
        })

        # Per-category metrics
        for category in results["metrics"]["categories"]:
            cat_metrics = results["metrics"]["categories"][category]
            total = cat_metrics["total_examples"]
            correct = cat_metrics["total_correct"]
            
            cat_metrics.update({
                "accuracy": correct / total if total > 0 else 0.0,
                "error_rate": 1 - (correct / total) if total > 0 else 1.0
            })
            
            # Per-choice accuracy for this category
            for choice in ANSWER_MAPPING.values():
                choice_stats = cat_metrics["per_choice_accuracy"][choice]
                choice_stats["accuracy"] = (
                    choice_stats["correct"] / choice_stats["total"]
                    if choice_stats["total"] > 0 else 0.0
                )

        # Calculate overall per-choice accuracy
        for choice in ANSWER_MAPPING.values():
            choice_stats = results["metrics"]["overall"]["per_choice_accuracy"][choice]
            choice_stats["accuracy"] = (
                choice_stats["correct"] / choice_stats["total"]
                if choice_stats["total"] > 0 else 0.0
            )

        # Performance metrics
        if latencies:
            latency_array = np.array(latencies)
            results["performance"]["latency_stats"].update({
                "mean": float(np.mean(latency_array)),
                "std": float(np.std(latency_array)),
                "min": float(np.min(latency_array)),
                "max": float(np.max(latency_array)),
                "p50": float(np.percentile(latency_array, 50)),
                "p90": float(np.percentile(latency_array, 90)),
                "p95": float(np.percentile(latency_array, 95)),
                "p99": float(np.percentile(latency_array, 99))
            })

        # Token throughput metrics
        total_time = results["performance"]["total_inference_time"]
        total_tokens = results["performance"]["total_tokens"]
        
        results["performance"].update({
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
            "average_tokens_per_example": total_tokens / total_processed if total_processed > 0 else 0.0,
            "average_latency_per_example": total_time / total_processed if total_processed > 0 else 0.0
        })

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

    return results

def run_benchmark(
    model_id: str,
    task_names: List[str],
    system_prompt: str = "You are a helpful AI assistant that answers multiple choice questions. Always answer with the letter (A, B, C, or D) of your choice.",
    num_few_shot: int = 5,
    max_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 42,
    log_dir: str = "logs",
    output_dir: str = "results",
    cache_dir: str = "task_cache",
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    batch_size: int = 8,
    dtype: Optional[str] = None,
    enforce_eager: Optional[bool] = None,
    num_samples: int = 0,
    cpu_offload_gb: float = 0,
    prefill_chunk_size: Optional[int] = None,
    kv_cache_dtype: Optional[str] = None,
    disable_bestpath: bool = False,
    max_num_seqs: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the benchmark on the MMLU dataset using vLLM with few-shot prompting."""
    from vllm import LLM, SamplingParams
    
    # Generate a unique run ID and setup logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(log_dir, run_id)
    logger.info(f"Starting benchmark run {run_id} with model {model_id}")
    
    # Initialize results structure
    category_results = {category: {"correct": 0, "total": 0} for category in task_names}
    
    # Configure model settings based on GPU capabilities
    gpu_capabilities = check_gpu_capabilities()
    memory_settings = optimize_memory_settings(gpu_capabilities, cpu_offload_gb)
    dtype = memory_settings["dtype"] if dtype is None else dtype
    enforce_eager = memory_settings["enforce_eager"] if enforce_eager is None else enforce_eager
    prefill_chunk_size = memory_settings["prefill_chunk_size"] if prefill_chunk_size is None else prefill_chunk_size
    
    # Initialize model kwargs
    model_kwargs = {
        "model": model_id,
        "gpu_memory_utilization": memory_settings["gpu_memory_utilization"],
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "seed": seed,
        "dtype": dtype,
        **({} if max_model_len is None else {"max_model_len": max_model_len}),
        **({} if not enforce_eager else {"enforce_eager": True}),
        **({} if cpu_offload_gb <= 0 else {"cpu_offload": True, "max_cpu_memory": f"{cpu_offload_gb}GiB"}),
        **({} if kv_cache_dtype is None else {"kv_cache_dtype": kv_cache_dtype}),
        **({} if prefill_chunk_size is None else {"prefill_chunk_size": prefill_chunk_size}),
        **({} if not disable_bestpath else {"disable_bestpath_optimization": True}),
        **({} if max_num_seqs is None else {"max_num_seqs": max_num_seqs})
    }
    
    # Load datasets with optimized settings
    try:
        datasets_dict = load_all_mmlu_categories(task_names, "test")
        if not datasets_dict:
            raise ValueError("No datasets were successfully loaded")
    except Exception as e:
        logger.error(f"Failed to load MMLU datasets: {e}")
        return {
            "error": f"Dataset loading failed: {str(e)}",
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Initialize model
    try:
        model = LLM(**model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "error": str(e),
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Prepare examples
    benchmark_examples = []
    few_shot_examples = []
    
    # Get few-shot examples if needed
    if num_few_shot > 0:
        for category, dataset in datasets_dict.items():
            if len(dataset) > num_few_shot:
                indices = np.random.choice(len(dataset), num_few_shot, replace=False)
                few_shot_examples.extend([dataset[int(i)] for i in indices])
    
    # Prepare benchmark examples
    for category, dataset in datasets_dict.items():
        examples = dataset
        if num_samples > 0:
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
            examples = [dataset[int(i)] for i in indices]
        benchmark_examples.extend([(category, ex) for ex in examples])
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Initialize results structure with all metrics
    results = {
        "run_id": run_id,
        "model_id": model_id,
        "run_config": {k: str(v) for k, v in model_kwargs.items()},
        "tasks": {category: [] for category in task_names},
        "metrics": {
            "overall": {
                "accuracy": 0.0,
                "total_correct": 0,
                "total_examples": 0,
                "confidence_score": 0.0,
                "calibration_error": 0.0,
                "per_choice_accuracy": {choice: {"correct": 0, "total": 0} for choice in ANSWER_MAPPING.values()}
            },
            "categories": {
                category: {
                    "accuracy": 0.0,
                    "total_correct": 0,
                    "total_examples": 0,
                    "confidence_score": 0.0,
                    "calibration_error": 0.0,
                    "per_choice_accuracy": {choice: {"correct": 0, "total": 0} for choice in ANSWER_MAPPING.values()}
                } for category in task_names
            }
        },
        "performance": {
            "total_tokens": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
            "latency_stats": {
                "mean": 0.0,
                "std": 0.0,
                "min": float('inf'),
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        }
    }
    
    # Run benchmark with batched processing
    from tqdm import tqdm
    total_examples = len(benchmark_examples)
    total_correct = 0
    total_processed = 0
    latencies = []
    
    with tqdm(total=total_examples, desc="MMLU Progress", ncols=100) as pbar:
        for batch_idx in range(0, total_examples, batch_size):
            batch_end = min(batch_idx + batch_size, total_examples)
            current_batch = benchmark_examples[batch_idx:batch_end]
            
            # Process batch
            outputs, batch_time = process_batch(model, current_batch, sampling_params, few_shot_examples, system_prompt)
            per_example_time = batch_time / len(current_batch)
            latencies.extend([per_example_time] * len(current_batch))
            
            # Process results
            for idx, (category, example) in enumerate(current_batch):
                output = outputs[idx]
                text = output.outputs[0].text
                predicted = extract_answer(text)
                correct = predicted and predicted.upper() == ANSWER_MAPPING[example["answer"]]
                
                if correct:
                    total_correct += 1
                    results["metrics"]["categories"][category]["total_correct"] += 1
                    results["metrics"]["overall"]["per_choice_accuracy"][ANSWER_MAPPING[example["answer"]]]["correct"] += 1
                    results["metrics"]["categories"][category]["per_choice_accuracy"][ANSWER_MAPPING[example["answer"]]]["correct"] += 1
                
                total_processed += 1
                results["metrics"]["categories"][category]["total_examples"] += 1
                results["metrics"]["overall"]["per_choice_accuracy"][ANSWER_MAPPING[example["answer"]]]["total"] += 1
                results["metrics"]["categories"][category]["per_choice_accuracy"][ANSWER_MAPPING[example["answer"]]]["total"] += 1
                
                # Store result details with confidence score
                confidence_score = 1.0 if correct else 0.0  # Simple binary confidence
                results["tasks"][category].append({
                    "question": example["question"],
                    "choices": example["choices"],
                    "correct_answer": ANSWER_MAPPING[example["answer"]],
                    "predicted": predicted,
                    "is_correct": correct,
                    "confidence_score": confidence_score,
                    "model_output": text,
                    "tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                    "inference_time": per_example_time
                })
                
                # Update performance metrics
                results["performance"]["total_tokens"] += len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
                results["performance"]["total_inference_time"] += per_example_time
            
            # Update progress
            pbar.update(len(current_batch))
            pbar.set_postfix({"Acc": f"{total_correct/total_processed:.2%}"})
    
    # Calculate all metrics properly
    results = calculate_metrics(results, total_correct, total_processed, latencies)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"results_{run_id}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Main function to run the MMLU benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM on MMLU with few-shot prompting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model-id", type=str, required=True,
                        help="Hugging Face model ID or local path")
    
    # Optional model configuration arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--system-prompt", type=str,
                      default="You are a helpful AI assistant that answers multiple choice questions. Always answer with the letter (A, B, C, or D) of your choice.",
                      help="System prompt for the model")
    model_group.add_argument("--max-tokens", type=int, default=32,
                      help="Maximum number of tokens to generate")
    model_group.add_argument("--temperature", type=float, default=0.0,
                      help="Sampling temperature")
    model_group.add_argument("--max-model-len", type=int, default=None,
                      help="Maximum sequence length for the model")
    model_group.add_argument("--dtype", type=str,
                      choices=["float16", "bfloat16", "float32", "auto"],
                      default="auto",
                      help="Datatype to use for model weights")
    model_group.add_argument("--kv-cache-dtype", type=str,
                      choices=["auto", "fp8", "fp16", "bf16"],
                      default="auto",
                      help="KV cache data type")
    
    # MMLU specific arguments
    mmlu_group = parser.add_argument_group("MMLU Configuration")
    mmlu_group.add_argument("--categories", type=str, nargs="+",
                     choices=MMLU_CATEGORIES + ["all"],
                     default=["all"],
                     help="MMLU categories to evaluate")
    mmlu_group.add_argument("--num-samples", type=int, default=0,
                     help="Number of samples per category (0 for all)")
    mmlu_group.add_argument("--num-few-shot", type=int, default=5,
                     help="Number of few-shot examples to use (default: 5)")
    
    # Hardware utilization arguments
    hw_group = parser.add_argument_group("Hardware Utilization")
    hw_group.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                    help="GPU memory utilization for vLLM (0.0-1.0)")
    hw_group.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism")
    hw_group.add_argument("--cpu-offload-gb", type=float, default=0,
                    help="Amount of GPU memory to offload to CPU (in GB)")
    hw_group.add_argument("--enforce-eager", action="store_true",
                    help="Enforce eager mode (disable CUDA graph)")
    hw_group.add_argument("--disable-cuda-graphs", action="store_true",
                    help="Explicitly disable CUDA graphs")
    hw_group.add_argument("--max-num-seqs", type=int, default=None,
                    help="Maximum number of concurrent sequences")
    hw_group.add_argument("--prefill-chunk-size", type=int, default=None,
                    help="Chunk size for prefill phase (if supported)")
    hw_group.add_argument("--disable-bestpath", action="store_true",
                    help="Disable bestpath scheduling optimization")
    
    # Benchmark configuration arguments
    bench_group = parser.add_argument_group("Benchmark Configuration")
    bench_group.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    bench_group.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for processing examples")
    
    # Output configuration arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--log-dir", type=str, default="logs",
                       help="Directory to save logs")
    output_group.add_argument("--output-dir", type=str, default="results",
                       help="Directory to save results")
    output_group.add_argument("--debug", action="store_true",
                       help="Enable debug mode with additional logging")
    output_group.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity of output")
    
    args = parser.parse_args()
    
    # Initialize logger
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(args.log_dir, run_id)
    
    # Set environment variables based on args
    if args.debug:
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    
    if args.quiet:
        os.environ["VLLM_DISABLE_TQDM"] = "1"
        os.environ["VLLM_LOG_LEVEL"] = "ERROR"
        os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    
    # Process categories
    if "all" in args.categories:
        categories = MMLU_CATEGORIES
    else:
        categories = args.categories
    
    # Handle dtype and other parameter configuration
    dtype = None if args.dtype == "auto" else args.dtype
    kv_cache_dtype = None if args.kv_cache_dtype == "auto" else args.kv_cache_dtype
    enforce_eager = args.enforce_eager or args.disable_cuda_graphs
    
    # Add logging for prefill chunk size
    if args.prefill_chunk_size is not None:
        logger.info(f"Using prefill chunk size: {args.prefill_chunk_size}")
        logger.info("Enabled chunked prefill for better throughput")
    
    # Run the benchmark
    start_time = time.time()
    print(f"Starting MMLU benchmark for model: {args.model_id}")
    print(f"Evaluating {len(categories)} categories")
    
    try:
        results = run_benchmark(
            model_id=args.model_id,
            task_names=categories,
            system_prompt=args.system_prompt,
            num_samples=args.num_samples,
            num_few_shot=args.num_few_shot,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            cpu_offload_gb=args.cpu_offload_gb,
            dtype=dtype,
            enforce_eager=enforce_eager,
            batch_size=args.batch_size,
            prefill_chunk_size=args.prefill_chunk_size,
            kv_cache_dtype=kv_cache_dtype,
            disable_bestpath=args.disable_bestpath,
            max_num_seqs=args.max_num_seqs
        )
    except Exception as e:
        import traceback
        logger.error(f"Benchmark failed with error: {e}")
        logger.error(traceback.format_exc())
        return 1
    
    # Print final summary
    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.2f}s")
    print("\nBenchmark Summary:")
    print(f"Model: {args.model_id}")
    print(f"Run ID: {results.get('run_id', 'unknown')}")
    
    # Check for errors first
    if "error" in results:
        print(f"\nERROR during benchmark: {results['error']}")
        return 1
    
    # Access metrics correctly from the results dictionary
    if "metrics" in results and "overall" in results["metrics"]:
        metrics = results["metrics"]["overall"]
        total_examples = metrics.get("total_examples", 0)
        accuracy = metrics.get("accuracy", 0.0)
        confidence_score = metrics.get("confidence_score", 0.0)
        calibration_error = metrics.get("calibration_error", 0.0)
        
        print(f"Total examples: {total_examples}")
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Confidence score: {confidence_score:.4f}")
        print(f"Calibration error: {calibration_error:.4f}")
    else:
        # For summary results from the function directly
        total_examples = results.get("total_examples", 0)
        accuracy = results.get("overall_accuracy", 0.0)
        print(f"Total examples: {total_examples}")
        print(f"Overall accuracy: {accuracy:.4f}")
    
    print(f"Total time: {total_time:.2f}s")
    
    # Print category-wise results if available
    print("\nCategory-wise Results:")
    category_results = None
    
    if "metrics" in results and "categories" in results["metrics"]:
        category_results = results["metrics"]["categories"]
    elif "category_results" in results:
        category_results = results["category_results"]
    elif "category_detailed_metrics" in results:
        category_results = results["category_detailed_metrics"]
    
    if category_results:
        for category, cat_result in sorted(category_results.items()):
            if isinstance(cat_result, dict):
                accuracy = cat_result.get("accuracy", 0.0)
                total = cat_result.get("total_examples", cat_result.get("total", 0))
                correct = cat_result.get("total_correct", cat_result.get("correct", 0))
                print(f"{category}: {accuracy:.4f} ({correct}/{total})")
    else:
        print("No category results available")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 