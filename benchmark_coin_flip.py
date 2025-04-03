#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any
import torch
import requests
from pathlib import Path
import numpy as np
import subprocess
from functools import lru_cache
from tqdm import tqdm

# Suppress warnings and configure environment variables upfront
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"  # Disable vLLM's progress bar
os.environ["VLLM_DISABLE_TQDM"] = "1"  # Disable vLLM's tqdm
warnings.filterwarnings('ignore', category=Warning)

# GPU model to compute capability mapping
GPU_COMPUTE_MAP = {
    "V100": 7.0,    # Volta
    "P100": 6.0,    # Pascal
    "P40": 6.1,     # Pascal
    "P4": 6.1,      # Pascal
    "T4": 7.5,      # Turing
    "A100": 8.0,    # Ampere
    "A40": 8.6,     # Ampere
    "A30": 8.0,     # Ampere
    "A10": 8.6,     # Ampere
    "A10G": 8.6,    # Ampere
    "A6000": 8.6,   # Ampere
    "RTX 3090": 8.6,  # Ampere
    "RTX 3080": 8.6,  # Ampere
    "RTX 3070": 8.6,  # Ampere
    "RTX 3060": 8.6,  # Ampere
    "RTX 2080": 7.5,  # Turing
    "RTX 2070": 7.5,  # Turing
    "RTX 2060": 7.5,  # Turing
    "GTX 1080": 6.1,  # Pascal
    "GTX 1070": 6.1,  # Pascal
    "GTX 1060": 6.1,  # Pascal
    "H100": 9.0,     # Hopper
    "L40": 8.9,      # Ada Lovelace
    "L4": 8.9,       # Ada Lovelace
    "RTX 4090": 8.9,  # Ada Lovelace
    "RTX 4080": 8.9,  # Ada Lovelace
    "RTX 4070": 8.9,  # Ada Lovelace
    "RTX 4060": 8.9,  # Ada Lovelace
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
        
    # Get device information
    for i in range(capabilities["num_gpus"]):
        device_name = torch.cuda.get_device_name(i)
        capabilities["device_names"].append(device_name)
        
        # Get compute capability using nvidia-smi first
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
        except (subprocess.SubprocessError, ValueError, IndexError, TimeoutError):
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
                # Use a context manager to handle potential errors and cleanup
                with torch.cuda.device(i):
                    test_tensor = torch.zeros(1, dtype=torch.bfloat16, device=f"cuda:{i}")
                    del test_tensor  # Clean up
                    capabilities["has_bf16"] = True
            except (RuntimeError, TypeError):
                pass
    
    # Determine if eager attention should be used
    min_compute_capability = min(capabilities["compute_capabilities"]) if capabilities["compute_capabilities"] else 0
    capabilities["should_use_eager_attention"] = min_compute_capability < 8.0
    
    return capabilities

def download_dataset(output_path: str = "data/coin_flip_4.json") -> str:
    """Download the coin flip dataset from GitHub if it doesn't exist locally."""
    if os.path.exists(output_path):
        return output_path
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # GitHub raw content URL
    url = "https://raw.githubusercontent.com/sileix/chain-of-draft/main/data/coin_flip_4.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        print(f"Successfully downloaded dataset to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """Setup detailed logging for the benchmark run."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"benchmark_coin_flip_{run_id}.log")
    
    logger = logging.getLogger("vllm_benchmark")
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

def load_coin_flip_dataset(file_path: str) -> List[Dict]:
    """Load the coin flip dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['examples']

def format_few_shot_examples(examples: List[Dict], system_prompt: str, model: 'LLM') -> str:
    """Format few-shot examples using the model's chat template."""
    # Create conversation messages
    conversation = [{"role": "system", "content": system_prompt}]
    
    for example in examples:
        # Add user question and assistant response
        conversation.append({"role": "user", "content": example["question"]})
        conversation.append({"role": "assistant", "content": example["answer"]})
    
    # Format using the model's tokenizer chat template if available
    try:
        formatted_prompt = model.get_tokenizer().apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return formatted_prompt
    except (AttributeError, ValueError):
        # Fallback to manual formatting
        formatted_prompt = system_prompt + "\n\n"
        for i in range(0, len(conversation)-1, 2):
            if i+1 < len(conversation):
                user_msg = conversation[i+1]["content"]
                assistant_msg = conversation[i+2]["content"] if i+2 < len(conversation) else ""
                formatted_prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        return formatted_prompt

def extract_answer(response: str) -> Optional[str]:
    """Extract Yes/No answer from the model's response."""
    response = response.strip().lower()
    if "yes" in response:
        return "Yes"
    elif "no" in response:
        return "No"
    return None

def run_benchmark(
    model_id: str,
    dataset_path: str,
    system_prompt: str,
    num_samples: int = 0,
    num_few_shot: int = 5,
    max_tokens: int = 128,
    temperature: float = 0.0,
    seed: int = 42,
    log_dir: str = "logs",
    output_dir: str = "results",
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    cpu_offload_gb: float = 0,
    dtype: Optional[str] = None,
    enforce_eager: Optional[bool] = None,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Run the benchmark on the coin flip dataset using vLLM."""
    # Import vLLM only when needed
    from vllm import LLM, SamplingParams
    
    # Generate a unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    logger = setup_logging(log_dir, run_id)
    logger.info(f"Starting coin flip benchmark run {run_id} with model {model_id}")
    
    # Handle dtype
    if dtype == "auto":
        # Check if GPU supports BF16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = "bfloat16"
            logger.info("Using BFloat16 precision (GPU supports BF16)")
        else:
            dtype = "float16"
            logger.info("Using Float16 precision (GPU does not support BF16)")
    elif dtype is None:
        dtype = "float16"  # Default to float16 if not specified
        logger.info("Using default Float16 precision")

    # Load dataset
    try:
        dataset = load_coin_flip_dataset(dataset_path)
        logger.info(f"Loaded {len(dataset)} examples from coin flip dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {
            "error": f"Dataset loading failed: {str(e)}",
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Select examples for benchmark
    np.random.seed(seed)
    
    if num_samples > 0 and num_samples < len(dataset):
        selected_indices = np.random.choice(len(dataset), num_samples, replace=False)
        benchmark_examples = [dataset[int(i)] for i in selected_indices]
        remaining_indices = list(set(range(len(dataset))) - set(selected_indices))
        shot_indices = np.random.choice(remaining_indices, num_few_shot, replace=False)
    else:
        benchmark_examples = dataset
        shot_indices = np.random.choice(len(dataset), num_few_shot, replace=False)
    
    few_shot_examples = [dataset[int(i)] for i in shot_indices]
    
    # Initialize model
    start_time = time.time()
    try:
        model = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype=dtype,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            seed=seed
        )
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
    
    # Format few-shot examples
    few_shot_prompt = format_few_shot_examples(few_shot_examples, system_prompt, model)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Results storage with enhanced granularity
    results = {
        "run_id": run_id,
        "model_id": model_id,
        "run_config": {
            "system_prompt": system_prompt,
            "num_samples": num_samples,
            "num_few_shot": num_few_shot,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "batch_size": batch_size,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len
        },
        "hardware_info": check_gpu_capabilities(),
        "timing": {
            "start_time": datetime.now().isoformat(),
            "model_load_time": model_load_time
        },
        "examples": [],
        "metrics": {
            "correct_count": 0,
            "total_examples": 0,
            "accuracy": 0.0
        },
        "performance": {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0
        }
    }
    
    try:
        # Create progress bar for overall progress
        total_examples = len(benchmark_examples)
        progress_bar = tqdm(
            total=total_examples,
            desc="Coin Flip Progress",
            leave=True,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        progress_bar.set_postfix({"Acc": 0.0})  # Initialize accuracy with proper format
        
        total_correct = 0
        total_processed = 0
        
        # Process examples in batches
        for batch_idx in range(0, len(benchmark_examples), batch_size):
            batch_end = min(batch_idx + batch_size, len(benchmark_examples))
            current_batch = benchmark_examples[batch_idx:batch_end]
            batch_size_actual = len(current_batch)
            
            # Select few-shot examples for this batch
            if num_few_shot > 0:
                available_examples = [ex for i, ex in enumerate(benchmark_examples) 
                                   if i < batch_idx or i >= batch_end]
                few_shot_examples = np.random.choice(
                    available_examples, 
                    min(num_few_shot, len(available_examples)), 
                    replace=False
                ).tolist()
            else:
                few_shot_examples = []

            # Format few-shot examples
            few_shot_prompt = format_few_shot_examples(few_shot_examples, system_prompt, model)
            
            # Prepare prompts for the batch
            prompts = []
            for example in current_batch:
                if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "apply_chat_template"):
                    conversation = [{"role": "system", "content": system_prompt}]
                    
                    # Add few-shot examples
                    for fs_example in few_shot_examples:
                        conversation.append({"role": "user", "content": fs_example["question"]})
                        conversation.append({"role": "assistant", "content": fs_example["answer"]})
                    
                    # Add current question
                    conversation.append({"role": "user", "content": example["question"]})
                    
                    prompt = model.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt = few_shot_prompt + f"User: {example['question']}\nAssistant:"
                
                prompts.append(prompt)
            
            # Get model responses
            batch_start_time = time.time()
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params)
            batch_inference_time = time.time() - batch_start_time
            
            # Process results
            for idx in range(batch_size_actual):
                example = current_batch[idx]
                output = outputs[idx]
                
                # Get generated text and extract answer
                generated_text = output.outputs[0].text
                predicted_answer = extract_answer(generated_text)
                
                # Calculate correctness
                is_correct = predicted_answer == example["answer"]
                if is_correct:
                    results["metrics"]["correct_count"] += 1
                
                # Store detailed example results
                example_result = {
                    "example_id": batch_idx + idx,
                    "question": example["question"],
                    "ground_truth": example["answer"],
                    "few_shot_examples": [
                        {
                            "question": fs_ex["question"],
                            "answer": fs_ex["answer"]
                        } for fs_ex in few_shot_examples
                    ],
                    "full_prompt": prompts[idx],
                    "model_output": generated_text,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
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
                
                results["examples"].append(example_result)
                results["metrics"]["total_examples"] += 1
                
                # Update performance metrics
                results["performance"]["total_tokens"] += example_result["metrics"]["total_tokens"]
                results["performance"]["total_prompt_tokens"] += example_result["metrics"]["prompt_tokens"]
                results["performance"]["total_completion_tokens"] += example_result["metrics"]["completion_tokens"]
                results["performance"]["total_inference_time"] += example_result["metrics"]["inference_time"]
            
            # Update progress and accuracy
            total_processed += batch_size_actual
            total_correct += sum(1 for idx in range(batch_size_actual) 
                               if example_result["is_correct"])
            current_accuracy = total_correct / total_processed if total_processed > 0 else 0.0
            
            progress_bar.update(batch_size_actual)
            progress_bar.set_postfix({"Acc": f"{current_accuracy:.2%}"})
        
        progress_bar.close()
        
        # Calculate final metrics
        if results["metrics"]["total_examples"] > 0:
            results["metrics"]["accuracy"] = results["metrics"]["correct_count"] / results["metrics"]["total_examples"]
            results["performance"]["tokens_per_second"] = results["performance"]["total_tokens"] / results["performance"]["total_inference_time"]
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_results_file = os.path.join(output_dir, f"detailed_results_{run_id}.json")
        with open(detailed_results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to {detailed_results_file}")
        
        # Create and save summary (without example-level details)
        summary = {
            "run_id": run_id,
            "model_id": model_id,
            "run_config": results["run_config"],
            "hardware_info": results["hardware_info"],
            "timing": results["timing"],
            "metrics": results["metrics"],
            "performance": results["performance"]
        }
        
        summary_file = os.path.join(output_dir, f"summary_{run_id}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {summary_file}")
        
        return summary
            
    except Exception as e:
        logger.error(f"Error during benchmarking: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save partial results if available
        if results["examples"]:
            logger.info(f"Saving partial results from {len(results['examples'])} examples")
            return results
        else:
            return {
                "error": str(e), 
                "run_id": run_id,
                "model_id": model_id,
                "model_load_time": model_load_time,
                "timestamp": datetime.now().isoformat(),
                "run_config": results["run_config"]
            }

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM on coin flip dataset with few-shot prompting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--model-id", type=str, required=True,
                        help="Hugging Face model ID or local path")
    parser.add_argument("--dataset-path", type=str, default="data/coin_flip_4.json",
                        help="Path to the coin flip dataset JSON file. Will be downloaded if not found.")
    
    # Optional model configuration
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful AI assistant that determines if a coin is heads up after a series of flips.",
                        help="System prompt for the model")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--dtype", type=str,
                        choices=["float16", "bfloat16", "float32", "auto", None],
                        default="auto",
                        help="Datatype to use for model weights")
    
    # Hardware utilization
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM (0.0-1.0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Enforce eager mode (disable CUDA graph)")
    
    # Benchmark configuration
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of samples to benchmark (0 for all)")
    parser.add_argument("--num-few-shot", type=int, default=5,
                        help="Number of few-shot examples to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing examples")
    
    # Output configuration
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Download dataset if needed
    try:
        dataset_path = download_dataset(args.dataset_path)
    except Exception as e:
        print(f"Failed to get dataset: {e}")
        return 1
    
    # Run the benchmark
    start_time = time.time()
    print(f"Starting coin flip benchmark for model: {args.model_id}")
    
    try:
        results = run_benchmark(
            model_id=args.model_id,
            dataset_path=dataset_path,
            system_prompt=args.system_prompt,
            num_samples=args.num_samples,
            num_few_shot=args.num_few_shot,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=args.enforce_eager,
            batch_size=args.batch_size,
        )
        
        # Print final summary
        total_time = time.time() - start_time
        print(f"\nBenchmark completed in {total_time:.2f}s")
        print("\nBenchmark Summary:")
        print(f"Model: {args.model_id}")
        print(f"Run ID: {results.get('run_id', 'unknown')}")
        
        # Access metrics correctly from the results dictionary
        total_examples = results.get("metrics", {}).get("total_examples", 0)
        accuracy = results.get("metrics", {}).get("accuracy", 0.0)
        avg_inference_time = results.get("performance", {}).get("total_inference_time", 0.0) / total_examples
        throughput = total_examples / avg_inference_time
        
        print(f"Total examples: {total_examples}")
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Average inference time: {avg_inference_time:.2f}s")
        print(f"Throughput: {throughput:.2f} examples/s")
        print(f"Total time: {total_time:.2f}s")
        
        # Print error if any
        if "error" in results:
            print(f"\nERROR during benchmark: {results['error']}")
            
        return 0
        
    except Exception as e:
        import traceback
        print(f"Benchmark failed with error: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 