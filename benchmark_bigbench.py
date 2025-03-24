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
import random
import uuid

# Suppress warnings and configure environment variables upfront
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"  # Disable vLLM's progress bar
os.environ["VLLM_DISABLE_TQDM"] = "1"  # Disable vLLM's tqdm
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', message='.*GetPrototype.*')

# Defer imports to improve startup time
import numpy as np
import torch
import re
import subprocess
import io
import sys
import requests
from tqdm import tqdm

# Global lock for thread-safety
thread_lock = threading.RLock()

# Task URLs
TASK_URLS = {
    "date_understanding": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/task.json",
    "sports_understanding": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/sports_understanding/task.json"
}

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """Setup detailed logging for the benchmark run."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"benchmark_{run_id}.log")
    
    logger = logging.getLogger("bigbench_benchmark")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Set up file handler for all logs (DEBUG and above)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Set up console handler for important logs only (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters - detailed for file, minimal for console
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def download_task(task_name: str, cache_dir: str) -> Dict:
    """Download and cache a BIG-bench task."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{task_name}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Download if not cached
    url = TASK_URLS[task_name]
    response = requests.get(url)
    response.raise_for_status()
    task_data = response.json()
    
    # Cache the downloaded data
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(task_data, f, indent=2)
    
    return task_data

def format_prompt(example: Dict, task_name: str, few_shot_examples: Optional[List[Dict]] = None, system_prompt: str = "") -> str:
    """Format a prompt for the given task with optional few-shot examples."""
    prompt = system_prompt + "\n\n" if system_prompt else ""
    
    if task_name == "date_understanding":
        prompt += "Given the context, determine the correct date.\n\n"
    else:  # sports_understanding
        prompt += "Determine whether the following sports-related statement is plausible or implausible.\n\n"
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for idx, shot in enumerate(few_shot_examples):
            prompt += f"Example {idx + 1}:\n"
            prompt += f"Input: {shot['input']}\n"
            prompt += f"Choices:\n"
            for i, choice in enumerate(shot['target_scores'].keys()):
                prompt += f"{chr(65 + i)}. {choice}\n"
            correct_answer = max(shot['target_scores'].items(), key=lambda x: x[1])[0]
            prompt += f"Answer: {correct_answer}\n\n"
    
    # Add the actual question
    prompt += f"Now, please answer this question:\n"
    prompt += f"Input: {example['input']}\n"
    prompt += f"Choices:\n"
    for i, choice in enumerate(example['target_scores'].keys()):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer: "
    
    return prompt

def extract_answer(response: str, choices: List[str]) -> Optional[str]:
    """Extract the answer from the model's response."""
    if not response or not isinstance(response, str):
        return None
    
    # Clean up response
    response = response.strip().upper()
    
    # Try to find the answer in various formats
    patterns = [
        r'\b([ABCD])\b',  # Single letter answer
        r'(?:ANSWER|ANS|CHOICE)(?:\s*(?:IS|:))?\s*([ABCD])\b',  # "Answer: X" format
        r'(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE)\s+(?:IS\s+)?([ABCD])\b'  # More verbose formats
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # If no direct letter match, try to match the text of the choice
    response_lower = response.lower()
    for idx, choice in enumerate(choices):
        if choice.lower() in response_lower:
            return chr(65 + idx)
    
    return None

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
        
    # Get device information
    for i in range(capabilities["num_gpus"]):
        device_name = torch.cuda.get_device_name(i)
        capabilities["device_names"].append(device_name)
        
        # Try to determine compute capability
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            if result.stdout.strip():
                compute_capability = float(result.stdout.strip().split('\n')[i])
                capabilities["compute_capabilities"].append(compute_capability)
                if compute_capability >= 8.0:
                    capabilities["has_bf16"] = True
        except:
            # Default to conservative estimates if nvidia-smi fails
            capabilities["compute_capabilities"].append(7.0)
    
    # Determine if eager attention should be used
    min_compute_capability = min(capabilities["compute_capabilities"]) if capabilities["compute_capabilities"] else 0
    capabilities["should_use_eager_attention"] = min_compute_capability < 8.0
    
    return capabilities

def run_benchmark(
    model_id: str,
    task_names: List[str],
    system_prompt: str = "You are a helpful AI assistant that answers questions accurately and truthfully.",
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
) -> Dict[str, Any]:
    """Run the benchmark on specified BIG-bench tasks."""
    # Import vLLM only when needed
    from vllm import LLM, SamplingParams
    
    # Generate a unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    logger = setup_logging(log_dir, run_id)
    logger.info(f"Starting benchmark run {run_id} with model {model_id}")
    
    # Check GPU capabilities
    gpu_capabilities = check_gpu_capabilities()
    logger.info(f"GPU capabilities: {json.dumps(gpu_capabilities, indent=2)}")
    
    # Determine appropriate dtype if not specified
    if dtype is None:
        if gpu_capabilities["has_bf16"]:
            dtype = "bfloat16"
            logger.info("Using BFloat16 precision (GPU supports BF16)")
        else:
            dtype = "float16"
            logger.info("Using Float16 precision (GPU does not support BF16)")
    
    # Set eager attention if not specified
    if enforce_eager is None:
        enforce_eager = gpu_capabilities["should_use_eager_attention"]
        logger.info(f"Auto-configuring eager mode: {enforce_eager}")
    
    # Initialize model
    model_kwargs = {
        "model": model_id,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "seed": seed,
        "dtype": dtype,
    }
    
    if max_model_len is not None:
        model_kwargs["max_model_len"] = max_model_len
    
    if enforce_eager:
        model_kwargs["enforce_eager"] = True
    
    # Load model
    logger.info(f"Initializing vLLM with model {model_id}")
    start_time = time.time()
    try:
        model = LLM(**model_kwargs)
        model_load_time = time.time() - start_time
        logger.info(f"Model loaded in {model_load_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "error": str(e),
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Initialize results structure
    results = {
        "run_id": run_id,
        "model_id": model_id,
        "run_config": {
            "num_few_shot": num_few_shot,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "batch_size": batch_size,
            "dtype": dtype,
            "enforce_eager": enforce_eager
        },
        "hardware_info": check_gpu_capabilities(),
        "timing": {
            "start_time": datetime.now().isoformat(),
            "end_time": None
        },
        "tasks": {},
        "overall_metrics": {
            "total_correct": 0,
            "total_examples": 0,
            "average_accuracy": 0.0,
            "total_tokens": 0,
            "total_time": 0.0,
            "average_tokens_per_second": 0.0
        }
    }
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Process each task
    for task_name in task_names:
        logger.info(f"\nProcessing task: {task_name}")
        
        try:
            # Load task data
            task_data = download_task(task_name, cache_dir)
            examples = task_data["examples"]
            
            # Sample examples if num_samples specified
            if num_samples > 0 and num_samples < len(examples):
                examples = random.sample(examples, num_samples)
            
            task_results = {
                "metadata": {
                    "task_name": task_name,
                    "num_examples": len(examples),
                    "few_shot_count": num_few_shot
                },
                "examples": [],
                "metrics": {
                    "correct": 0,
                    "total": 0,
                    "accuracy": 0.0,
                    "average_score": 0.0
                },
                "performance": {
                    "total_tokens": 0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_inference_time": 0.0,
                    "tokens_per_second": 0.0
                }
            }
            
            # Create progress bar for overall progress
            total_examples = len(examples)
            progress_bar = tqdm(
                total=total_examples,
                desc=f"BIG-bench Progress",
                leave=True,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )
            progress_bar.set_postfix({"Acc": 0.0})
            
            # Process examples in batches
            for batch_idx in range(0, total_examples, batch_size):
                batch_size_actual = min(batch_size, total_examples - batch_idx)
                current_batch = examples[batch_idx:batch_idx + batch_size_actual]
                
                # Prepare prompts for the batch
                prompts = []
                few_shot_examples = []
                if num_few_shot > 0:
                    # Select random few-shot examples (excluding current batch)
                    available_examples = examples[:batch_idx] + examples[batch_idx + batch_size_actual:]
                    if len(available_examples) >= num_few_shot:
                        few_shot_examples = random.sample(available_examples, num_few_shot)
                
                for example in current_batch:
                    prompt = format_prompt(example, task_name, few_shot_examples, system_prompt)
                    prompts.append(prompt)
                
                # Get model responses
                batch_start_time = time.time()
                outputs = model.generate(
                    prompts=prompts,
                    sampling_params=SamplingParams(
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
                batch_inference_time = time.time() - batch_start_time
                
                # Process responses
                batch_correct = 0
                for idx, (example, output) in enumerate(zip(current_batch, outputs)):
                    response = output.outputs[0].text
                    choices = list(example["target_scores"].keys())
                    predicted_answer = extract_answer(response, choices)
                    
                    # Calculate score
                    score = 0.0
                    if predicted_answer:
                        answer_idx = ord(predicted_answer) - ord('A')
                        if 0 <= answer_idx < len(choices):
                            choice = choices[answer_idx]
                            score = example["target_scores"][choice]
                            if score > 0.5:
                                task_results["metrics"]["correct"] += 1
                                batch_correct += 1
                    
                    # Store detailed example results
                    example_result = {
                        "example_id": batch_idx + idx,
                        "input": example["input"],
                        "choices": choices,
                        "target_scores": example["target_scores"],
                        "few_shot_examples": [
                            {
                                "input": fs_ex["input"],
                                "choices": list(fs_ex["target_scores"].keys()),
                                "target_scores": fs_ex["target_scores"]
                            } for fs_ex in few_shot_examples
                        ],
                        "full_prompt": prompts[idx],
                        "model_output": response,
                        "predicted_answer": predicted_answer,
                        "score": score,
                        "metrics": {
                            "prompt_tokens": len(output.prompt_token_ids),
                            "completion_tokens": len(output.outputs[0].token_ids),
                            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
                            "inference_time": batch_inference_time / batch_size_actual,
                            "tokens_per_second": (len(output.prompt_token_ids) + len(output.outputs[0].token_ids)) / (batch_inference_time / batch_size_actual)
                        },
                        "batch_info": {
                            "batch_size": batch_size_actual,
                            "batch_inference_time": batch_inference_time,
                            "batch_index": batch_idx // batch_size
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    task_results["examples"].append(example_result)
                    task_results["metrics"]["total"] += 1
                    
                    # Update performance metrics
                    task_results["performance"]["total_tokens"] += example_result["metrics"]["total_tokens"]
                    task_results["performance"]["total_prompt_tokens"] += example_result["metrics"]["prompt_tokens"]
                    task_results["performance"]["total_completion_tokens"] += example_result["metrics"]["completion_tokens"]
                    task_results["performance"]["total_inference_time"] += example_result["metrics"]["inference_time"]
                
                # Update progress bar
                current_accuracy = batch_correct / batch_size_actual
                progress_bar.update(batch_size_actual)
                progress_bar.set_postfix({"Acc": f"{current_accuracy:.2%}"})
            
            progress_bar.close()
            
            # Calculate final task metrics
            if task_results["metrics"]["total"] > 0:
                task_results["metrics"]["accuracy"] = task_results["metrics"]["correct"] / task_results["metrics"]["total"]
                task_results["metrics"]["average_score"] = sum(ex["score"] for ex in task_results["examples"]) / task_results["metrics"]["total"]
                task_results["performance"]["tokens_per_second"] = task_results["performance"]["total_tokens"] / task_results["performance"]["total_inference_time"] if task_results["performance"]["total_inference_time"] > 0 else 0
            
            results["tasks"][task_name] = task_results
            
            # Update overall metrics
            results["overall_metrics"]["total_correct"] += task_results["metrics"]["correct"]
            results["overall_metrics"]["total_examples"] += task_results["metrics"]["total"]
            results["overall_metrics"]["total_tokens"] += task_results["performance"]["total_tokens"]
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results["tasks"][task_name] = {"error": str(e)}
    
    # Calculate final overall metrics
    total_examples = results["overall_metrics"]["total_examples"]
    if total_examples > 0:
        results["overall_metrics"]["average_accuracy"] = results["overall_metrics"]["total_correct"] / total_examples
        results["overall_metrics"]["total_time"] = time.time() - start_time
        results["overall_metrics"]["average_tokens_per_second"] = sum(
            task["performance"]["tokens_per_second"] 
            for task in results["tasks"].values() 
            if isinstance(task, dict) and "performance" in task
        ) / len([task for task in results["tasks"].values() if isinstance(task, dict) and "performance" in task])
    
    results["timing"]["end_time"] = datetime.now().isoformat()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_results_file = os.path.join(output_dir, f"detailed_results_{run_id}.json")
    with open(detailed_results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {detailed_results_file}")
    
    # Create and save summary
    summary = {
        "run_id": run_id,
        "model_id": model_id,
        "run_config": results["run_config"],
        "hardware_info": results["hardware_info"],
        "timing": results["timing"],
        "overall_metrics": results["overall_metrics"],
        "task_summaries": {
            task_name: {
                "metrics": task_data["metrics"],
                "performance": task_data["performance"]
            }
            for task_name, task_data in results["tasks"].items()
            if isinstance(task_data, dict) and "metrics" in task_data
        }
    }
    
    summary_file = os.path.join(output_dir, f"summary_{run_id}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_file}")

    return summary

def main():
    """Main function to run the BIG-bench benchmark."""
    parser = argparse.ArgumentParser(description="Run BIG-bench benchmark with vLLM")
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--tasks", type=str, nargs="+", choices=list(TASK_URLS.keys()), 
                       default=list(TASK_URLS.keys()), help="Tasks to run (default: all tasks)")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--cache-dir", type=str, default="task_cache", help="Directory to cache downloaded tasks")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, help="Maximum sequence length for the model")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--num-few-shot", type=int, default=5, help="Number of few-shot examples to use")
    parser.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32", "auto"],
                       default="auto", help="Data type for model weights")
    parser.add_argument("--enforce-eager", action="store_true", 
                       help="Enforce eager mode (disable CUDA graph)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional logging")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity of output")
    parser.add_argument("--system-prompt", type=str, 
                       default="You are a helpful AI assistant that answers questions accurately and truthfully.",
                       help="System prompt to use for the model")
    args = parser.parse_args()

    # Set environment variables based on args
    if args.debug:
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    
    if args.quiet:
        os.environ["VLLM_DISABLE_TQDM"] = "1"
        os.environ["VLLM_LOG_LEVEL"] = "ERROR"
        os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    
    # Run the benchmark
    start_time = time.time()
    print(f"Starting BIG-bench benchmark for model: {args.model_id}")
    print(f"Evaluating tasks: {', '.join(args.tasks)}")
    
    try:
        results = run_benchmark(
            model_id=args.model_id,
            task_names=args.tasks,
            system_prompt=args.system_prompt,
            num_few_shot=args.num_few_shot,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            batch_size=args.batch_size,
            dtype=None if args.dtype == "auto" else args.dtype,
            enforce_eager=args.enforce_eager,
            num_samples=args.num_samples,
        )
    except Exception as e:
        import traceback
        print(f"Benchmark failed with error: {e}")
        print(traceback.format_exc())
        return 1
    
    # Print final summary
    total_time = time.time() - start_time
    print(f"\nBenchmark completed in {total_time:.2f}s")
    print("\nBenchmark Summary:")
    print(f"Model: {args.model_id}")
    print(f"Run ID: {results.get('run_id', 'unknown')}")
    print(f"Number of few-shot examples: {args.num_few_shot}")
    
    # Access metrics correctly from the results dictionary
    overall_metrics = results.get("overall_metrics", {})
    total_examples = overall_metrics.get("total_examples", 0)
    avg_accuracy = overall_metrics.get("average_accuracy", 0.0)
    total_time_taken = overall_metrics.get("total_time", 0.0)
    avg_tokens_per_second = overall_metrics.get("average_tokens_per_second", 0.0)
    
    print(f"Total examples: {total_examples}")
    print(f"Overall accuracy: {avg_accuracy:.4f}")
    print(f"Average tokens/s: {avg_tokens_per_second:.2f}")
    print(f"Total time: {total_time_taken:.2f}s")
    
    # Print per-task results
    print("\nTask-wise Results:")
    for task_name, task_result in results.get("tasks", {}).items():
        if "error" in task_result:
            print(f"{task_name}: ERROR - {task_result['error']}")
        else:
            metrics = task_result.get("metrics", {})
            print(f"{task_name}:")
            print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            print(f"  Average Score: {metrics.get('average_score', 0.0):.4f}")
            print(f"  Accuracy: {task_result['metrics']['accuracy']:.4f}")
            print(f"  Average Score: {task_result['metrics']['average_score']:.4f}")
            print(f"  Correct: {task_result['metrics']['correct']}/{task_result['metrics']['total']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 