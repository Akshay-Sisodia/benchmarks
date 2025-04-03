#!/usr/bin/env python3

#------------------------------------------------------------------------------
# Standard Library Imports
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Third-Party Imports
#------------------------------------------------------------------------------
import numpy as np
import torch
import re
import subprocess
import io
import sys
import requests
from tqdm import tqdm

#------------------------------------------------------------------------------
# Environment Setup
#------------------------------------------------------------------------------

# Suppress warnings and configure environment variables upfront
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"  # Disable vLLM's progress bar
os.environ["VLLM_DISABLE_TQDM"] = "1"  # Disable vLLM's tqdm
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', message='.*GetPrototype.*')

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

# Task URLs for BIG-bench tasks
TASK_URLS = {
    "date_understanding": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/task.json",
    "sports_understanding": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/sports_understanding/task.json"
}

# GPU model to compute capability mapping
GPU_COMPUTE_MAP = {
    # Volta
    "V100": 7.0,
    
    # Pascal
    "P100": 6.0,
    "P40": 6.1,
    "P4": 6.1,
    
    # Turing
    "T4": 7.5,
    "RTX 2080": 7.5,
    "RTX 2070": 7.5,
    "RTX 2060": 7.5,
    
    # Ampere
    "A100": 8.0,
    "A40": 8.6,
    "A30": 8.0,
    "A10": 8.6,
    "A10G": 8.6,
    "A6000": 8.6,
    "RTX 3090": 8.6,
    "RTX 3080": 8.6,
    "RTX 3070": 8.6,
    "RTX 3060": 8.6,
    
    # Pascal (older)
    "GTX 1080": 6.1,
    "GTX 1070": 6.1,
    "GTX 1060": 6.1,
    
    # Hopper
    "H100": 9.0,
    
    # Ada Lovelace
    "L40": 8.9,
    "L4": 8.9,
    "RTX 4090": 8.9,
    "RTX 4080": 8.9,
    "RTX 4070": 8.9,
    "RTX 4060": 8.9,
}

# Global lock for thread-safety
thread_lock = threading.RLock()

#------------------------------------------------------------------------------
# Logging Setup
#------------------------------------------------------------------------------

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """
    Setup detailed logging for the benchmark run.
    
    Args:
        log_dir (str): Directory to store log files
        run_id (str): Unique identifier for this benchmark run
        
    Returns:
        logging.Logger: Configured logger instance
    """
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

#------------------------------------------------------------------------------
# Task Loading and Management
#------------------------------------------------------------------------------

def download_task(task_name: str, cache_dir: str) -> Dict:
    """
    Download and cache a BIG-bench task.
    
    Args:
        task_name (str): Name of the task to download
        cache_dir (str): Directory to cache downloaded tasks
        
    Returns:
        Dict: The loaded task data
        
    Raises:
        Exception: If download or caching fails
    """
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

#------------------------------------------------------------------------------
# Prompt Formatting and Answer Processing
#------------------------------------------------------------------------------

def format_prompt(
    example: Dict,
    task_name: str,
    few_shot_examples: Optional[List[Dict]] = None,
    system_prompt: str = ""
) -> str:
    """
    Format a prompt for the given task with optional few-shot examples.
    
    Args:
        example (Dict): The example to format
        task_name (str): Name of the task
        few_shot_examples (Optional[List[Dict]]): List of few-shot examples to include
        system_prompt (str): System prompt to prepend
        
    Returns:
        str: Formatted prompt string
    """
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
    """
    Extract the answer from the model's response.
    
    Args:
        response (str): The model's response text
        choices (List[str]): List of possible answer choices
        
    Returns:
        Optional[str]: The extracted answer or None if no valid answer found
    """
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

#------------------------------------------------------------------------------
# GPU and Hardware Management
#------------------------------------------------------------------------------

def check_gpu_capabilities() -> Dict[str, Any]:
    """
    Check GPU capabilities including BFloat16 support and compute capability.
    
    Returns:
        Dict containing:
        - has_cuda (bool): Whether CUDA is available
        - has_bf16 (bool): Whether BFloat16 is supported
        - num_gpus (int): Number of available GPUs
        - cuda_version (str): CUDA version if available
        - device_names (List[str]): Names of available GPU devices
        - compute_capabilities (List[float]): Compute capabilities of GPUs
        - should_use_eager_attention (bool): Whether to use eager attention mode
    """
    capabilities = {
        "has_cuda": torch.cuda.is_available(),
        "has_bf16": False,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_names": [],
        "compute_capabilities": [],
        "should_use_eager_attention": False,
    }

    if not capabilities["has_cuda"]:
        return capabilities

    # Get device information
    for i in range(capabilities["num_gpus"]):
        device_name = torch.cuda.get_device_name(i)
        capabilities["device_names"].append(device_name)

        # Get compute capability using nvidia-smi first
        compute_capability = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2
            )
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if i < len(lines):
                    compute_capability = float(lines[i].strip())
        except:
            pass

        # Fall back to name matching if nvidia-smi failed
        if compute_capability is None:
            for gpu_name, cc in GPU_COMPUTE_MAP.items():
                if gpu_name in device_name:
                    compute_capability = cc
                    break

            # Default if we couldn't determine
            if compute_capability is None:
                compute_capability = 7.0  # Conservative default

        capabilities["compute_capabilities"].append(compute_capability)

        # Check BF16 support (compute capability >= 8.0)
        if compute_capability >= 8.0:
            capabilities["has_bf16"] = True

        # Alternative check: try to create a BF16 tensor
        if not capabilities["has_bf16"]:
            try:
                with torch.cuda.device(i):
                    test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=f"cuda:{i}")
                    del test_tensor  # Clean up
                    capabilities["has_bf16"] = True
            except:
                pass

    # Determine if eager attention should be used
    min_compute_capability = min(capabilities["compute_capabilities"]) if capabilities["compute_capabilities"] else 0
    capabilities["should_use_eager_attention"] = min_compute_capability < 8.0

    return capabilities

#------------------------------------------------------------------------------
# Benchmark Core Logic
#------------------------------------------------------------------------------

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
    """
    Run the benchmark on specified BIG-bench tasks.
    
    Args:
        model_id (str): HuggingFace model ID or local path
        task_names (List[str]): List of BIG-bench tasks to evaluate
        system_prompt (str): System prompt for the model
        num_few_shot (int): Number of few-shot examples to use
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        seed (int): Random seed for reproducibility
        log_dir (str): Directory for logs
        output_dir (str): Directory for results
        cache_dir (str): Directory for caching tasks
        gpu_memory_utilization (float): GPU memory utilization (0-1)
        max_model_len (Optional[int]): Maximum sequence length
        tensor_parallel_size (int): Number of GPUs for tensor parallelism
        batch_size (int): Batch size for processing
        dtype (Optional[str]): Model dtype (float16, bfloat16, etc.)
        enforce_eager (Optional[bool]): Whether to enforce eager execution
        num_samples (int): Number of samples per task (0 for all)
        
    Returns:
        Dict[str, Any]: Benchmark results and metrics
    """
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

#------------------------------------------------------------------------------
# Command Line Interface
#------------------------------------------------------------------------------

def main():
    """
    Main function to run the BIG-bench benchmark from command line.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Run BIG-bench benchmark with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model ID or local path"
    )
    
    # Task selection arguments
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=list(TASK_URLS.keys()),
        default=list(TASK_URLS.keys()),
        help="Tasks to run (default: all tasks)"
    )
    task_group.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to evaluate (0 for all)"
    )
    
    # Model configuration arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    model_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    
    # Output configuration arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    output_group.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save logs"
    )
    output_group.add_argument(
        "--cache-dir",
        type=str,
        default="task_cache",
        help="Directory to cache downloaded tasks"
    )
    
    # Hardware utilization arguments
    hw_group = parser.add_argument_group("Hardware Utilization")
    hw_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization"
    )
    hw_group.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum sequence length for the model"
    )
    hw_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    hw_group.add_argument(
        "--num-few-shot",
        type=int,
        default=5,
        help="Number of few-shot examples to use"
    )
    hw_group.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32", "auto"],
        default="auto",
        help="Data type for model weights"
    )
    hw_group.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graph)"
    )
    
    # Debug and verbosity arguments
    debug_group = parser.add_argument_group("Debug Configuration")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    debug_group.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity of output"
    )
    debug_group.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant that answers questions accurately and truthfully.",
        help="System prompt to use for the model"
    )
    
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
            print(f"  Correct: {metrics['correct']}/{metrics['total']}")
    
    return 0

#------------------------------------------------------------------------------
# Entry Point
#------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main()) 