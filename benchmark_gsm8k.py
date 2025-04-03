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
import datasets

# Suppress warnings and configure environment variables upfront
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"  # Disable vLLM's progress bar
os.environ["VLLM_DISABLE_TQDM"] = "1"  # Disable vLLM's tqdm
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

# Defer imports to improve startup time
import numpy as np
import torch
import re
import subprocess
import io
import sys

# Global lock for thread-safety
nltk_lock = threading.RLock()
nltk_initialized = False


# Setup NLTK data path and download required resources
def setup_nltk(custom_data_dir=None):
    """Setup NLTK data path and download required resources efficiently."""
    global nltk_initialized

    with nltk_lock:
        if nltk_initialized:
            return True

        # Import NLTK only when needed
        import nltk

        # Create a directory for NLTK data
        nltk_data_dir = custom_data_dir or os.path.expanduser("~/nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)

        # Set NLTK data path
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)

        print(f"NLTK data directory set to: {nltk_data_dir}")

        # Required resources
        resources = [
            "punkt",
            "wordnet",
            "omw-1.4",
            "averaged_perceptron_tagger",
            "universal_tagset",
        ]

        # Download resources in parallel
        success = True
        with ThreadPoolExecutor(max_workers=min(5, len(resources))) as executor:
            futures = {
                executor.submit(
                    lambda r: nltk.download(r, download_dir=nltk_data_dir, quiet=True),
                    resource,
                ): resource
                for resource in resources
            }

            for future in futures:
                resource = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to download {resource}: {e}")
                    success = False

        # Special handling for punkt_tab
        try:
            nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
        except Exception:
            # Create the punkt_tab directory structure manually
            punkt_tab_dir = os.path.join(nltk_data_dir, "tokenizers", "punkt_tab")
            english_dir = os.path.join(punkt_tab_dir, "english")
            os.makedirs(english_dir, exist_ok=True)

            # Create a minimal punkt_tab file if it doesn't exist
            punkt_tab_file = os.path.join(english_dir, "punkt.tab")
            if not os.path.exists(punkt_tab_file):
                try:
                    with open(punkt_tab_file, "w") as f:
                        f.write(
                            ".\t.\tMr.\tMs.\tMrs.\tDr.\tProf.\tInc.\tCo.\tCorp.\tLtd.\tetc.\te.g.\ti.e.\tvs."
                        )
                except Exception as write_err:
                    print(f"Failed to create punkt_tab file: {write_err}")
                    success = False

        nltk_initialized = True
        return success


# Lazy-loaded imports and setup
def get_nltk():
    """Lazy-load NLTK only when needed."""
    import nltk

    if not nltk_initialized:
        setup_nltk()
    return nltk


@lru_cache(maxsize=1)
def get_rouge_scorer():
    """Lazy-load rouge_scorer with caching."""
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


# Set up logging
def setup_logging(log_dir: str, run_id: str) -> logging.Logger:
    """Setup detailed logging for the benchmark run."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"benchmark_{run_id}.log")

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
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Load GSM8K dataset
@lru_cache(maxsize=1)
def load_gsm8k(split: str = "test") -> "datasets.Dataset":
    """Load the GSM8K dataset with caching."""
    import datasets

    return datasets.load_dataset("gsm8k", "main")[split]


# Pre-compile regex patterns for extraction
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
DOLLAR_PATTERN = re.compile(r"\$\s*([-+]?\d*\.\d+|[-+]?\d+)(?:[^\d]*)$")
DOLLAR_NUMBERS_PATTERN = re.compile(r"\$\s*([-+]?\d*\.\d+|[-+]?\d+)")
ANSWER_PHRASE_PATTERNS = [
    re.compile(
        r"(?:answer|result|total|sum|cost)(?:\s+is)?\s*\$?\s*([-+]?\d*\.\d+|[-+]?\d+)"
    ),
    re.compile(
        r"(?:spend|pay|price|amount)(?:\s+of)?\s*\$?\s*([-+]?\d*\.\d+|[-+]?\d+)"
    ),
]

# New regex patterns based on LMSYS evaluation framework
STRICT_MATCH_PATTERN = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEXIBLE_EXTRACT_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


# Extract answer from model response for GSM8K using strict matching (#### format)
def extract_strict_answer(response: str) -> Optional[float]:
    """Extract the numerical answer from the model's response for GSM8K problems using strict matching.

    Looks for a pattern like '#### 123' and extracts the number.
    """
    if not response or not response.strip():
        return None

    # Look for '#### NUMBER' pattern
    match = STRICT_MATCH_PATTERN.search(response)
    if match:
        # Clean up the matched number (remove commas)
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None

    return None


# Extract answer from model response for GSM8K using flexible extraction
def extract_flexible_answer(response: str) -> Optional[float]:
    """Extract the numerical answer from the model's response for GSM8K problems using flexible extraction.

    Finds all numbers or dollar amounts in the text and takes the last one.
    """
    if not response or not response.strip():
        return None

    try:
        # Find all matches for the flexible pattern
        matches = FLEXIBLE_EXTRACT_PATTERN.findall(response)
        if not matches:
            return None

        # Flatten the list of tuples and remove empty strings
        matches = [group for match in matches for group in match if group]

        if not matches:
            return None

        # Take the last match
        last_match = matches[-1]

        # Clean up the matched string (remove $ and commas)
        last_match = last_match.replace("$", "").replace(",", "")

        try:
            return float(last_match)
        except ValueError:
            return None
    except Exception:
        # Return None for any unexpected errors
        return None


# Keep the original extract_answer for backward compatibility
def extract_answer(response: str) -> Optional[float]:
    """Extract the numerical answer from the model's response for GSM8K problems.

    First tries the strict pattern, then falls back to flexible extraction.
    """
    try:
        # Try strict matching first
        strict_answer = extract_strict_answer(response)
        if strict_answer is not None:
            return strict_answer

        # Fall back to flexible extraction
        return extract_flexible_answer(response)
    except Exception:
        # Return None for any unexpected errors
        return None


# Extract ground truth answer from GSM8K
def extract_ground_truth(answer: str) -> Optional[float]:
    """Extract the numerical answer from the ground truth for GSM8K problems."""
    if not answer:
        return None

    try:
        numbers = NUMBER_PATTERN.findall(answer)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None
    except Exception:
        # Return None for any unexpected errors
        return None


# Format five-shot multi-turn examples for GSM8K
def format_five_shot_multiturn(
    examples: List[Dict], system_prompt: str, model: "LLM"
) -> str:
    """Format five examples as a multi-turn conversation using the model's chat template."""
    # Create conversation messages
    conversation = [{"role": "system", "content": system_prompt}]

    for example in examples:
        # Add user question and assistant response
        conversation.append({"role": "user", "content": example["question"]})
        conversation.append({"role": "assistant", "content": example["answer"]})

    # Format using the model's tokenizer chat template if available
    try:
        formatted_prompt = model.get_tokenizer().apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt
    except (AttributeError, ValueError):
        # Fallback to manual formatting if chat template is not available
        formatted_prompt = system_prompt + "\n\n"
        for i in range(0, len(conversation) - 1, 2):
            if i + 1 < len(conversation):
                user_msg = conversation[i + 1]["content"]
                assistant_msg = (
                    conversation[i + 2]["content"] if i + 2 < len(conversation) else ""
                )
                formatted_prompt += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        return formatted_prompt


# Calculate BLEU, ROUGE, and METEOR scores
def calculate_nlg_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate NLG metrics: BLEU, ROUGE, and METEOR with better performance."""
    # Skip metric calculation if input is too short
    if len(prediction.strip()) < 5 or len(reference.strip()) < 5:
        return {
            "bleu": 0.0,
            "meteor": 0.0,
            "rouge1_precision": 0.0,
            "rouge1_recall": 0.0,
            "rouge1_fmeasure": 0.0,
            "rouge2_precision": 0.0,
            "rouge2_recall": 0.0,
            "rouge2_fmeasure": 0.0,
            "rougeL_precision": 0.0,
            "rougeL_recall": 0.0,
            "rougeL_fmeasure": 0.0,
            "error": "Input text too short for meaningful NLG metrics",
        }

    # Initialize empty metrics dictionary
    metrics = {
        "bleu": 0.0,
        "meteor": 0.0,
        "rouge1_precision": 0.0,
        "rouge1_recall": 0.0,
        "rouge1_fmeasure": 0.0,
        "rouge2_precision": 0.0,
        "rouge2_recall": 0.0,
        "rouge2_fmeasure": 0.0,
        "rougeL_precision": 0.0,
        "rougeL_recall": 0.0,
        "rougeL_fmeasure": 0.0,
    }

    try:
        # Import NLTK only when needed
        nltk = get_nltk()
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score

        # Tokenize the texts with better error handling
        try:
            pred_tokens = nltk.word_tokenize(prediction.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
        except Exception as tokenize_error:
            # Fallback tokenization
            try:
                pred_tokens = prediction.lower().split()
                ref_tokens = reference.lower().split()
            except Exception:
                metrics["error"] = f"Failed to tokenize text: {str(tokenize_error)}"
                return metrics

        # Calculate BLEU score
        try:
            metrics["bleu"] = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method1,
            )
        except Exception as bleu_error:
            metrics["error_bleu"] = str(bleu_error)

        # Calculate METEOR score
        try:
            metrics["meteor"] = meteor_score([ref_tokens], pred_tokens)
        except Exception as meteor_error:
            metrics["error_meteor"] = str(meteor_error)

        # Calculate ROUGE scores efficiently
        try:
            scorer = get_rouge_scorer()
            rouge_scores = scorer.score(reference, prediction)

            # Extract all ROUGE scores at once
            for metric in ["rouge1", "rouge2", "rougeL"]:
                if metric in rouge_scores:
                    score = rouge_scores[metric]
                    metrics[f"{metric}_precision"] = score.precision
                    metrics[f"{metric}_recall"] = score.recall
                    metrics[f"{metric}_fmeasure"] = score.fmeasure
        except Exception as rouge_error:
            metrics["error_rouge"] = str(rouge_error)

        return metrics

    except Exception as e:
        # Add error info to metrics but return valid structure
        metrics["error"] = str(e)
        return metrics


# GPU model to compute capability mapping
GPU_COMPUTE_MAP = {
    "V100": 7.0,  # Volta
    "P100": 6.0,  # Pascal
    "P40": 6.1,  # Pascal
    "P4": 6.1,  # Pascal
    "T4": 7.5,  # Turing
    "A100": 8.0,  # Ampere
    "A40": 8.6,  # Ampere
    "A30": 8.0,  # Ampere
    "A10": 8.6,  # Ampere
    "A10G": 8.6,  # Ampere
    "A6000": 8.6,  # Ampere
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
    "H100": 9.0,  # Hopper
    "L40": 8.9,  # Ada Lovelace
    "L4": 8.9,  # Ada Lovelace
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
                timeout=2,
            )
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
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
                    test_tensor = torch.zeros(
                        1, dtype=torch.bfloat16, device=f"cuda:{i}"
                    )
                    del test_tensor  # Clean up
                    capabilities["has_bf16"] = True
            except (RuntimeError, TypeError):
                pass

    # Determine if eager attention should be used
    min_compute_capability = (
        min(capabilities["compute_capabilities"])
        if capabilities["compute_capabilities"]
        else 0
    )
    capabilities["should_use_eager_attention"] = min_compute_capability < 8.0

    return capabilities


# Run benchmark
def run_benchmark(
    model_id: str,
    system_prompt: str,
    num_samples: int = 0,
    num_few_shot: int = 5,
    max_tokens: int = 512,
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
    """Run the benchmark on the GSM8K dataset using vLLM with few-shot prompting."""
    # Import vLLM only when needed
    from vllm import LLM, SamplingParams

    # Generate a unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logging
    logger = setup_logging(log_dir, run_id)
    logger.info(f"Starting benchmark run {run_id} with model {model_id}")
    logger.info(
        f"Parameters: num_samples={num_samples}, num_few_shot={num_few_shot}, "
        f"max_tokens={max_tokens}, temperature={temperature}, batch_size={batch_size}"
    )

    # Check GPU capabilities
    gpu_capabilities = check_gpu_capabilities()
    logger.info(f"GPU capabilities: {json.dumps(gpu_capabilities, indent=2)}")

    # Determine appropriate dtype and eager mode if not specified
    if dtype is None:
        if gpu_capabilities["has_bf16"]:
            dtype = "bfloat16"
            logger.info("Using BFloat16 precision (GPU supports BF16)")
        else:
            dtype = "float16"
            logger.info("Using Float16 precision (GPU does not support BF16)")

    # Set eager attention if not specified and GPU has compute capability < 8.0
    if enforce_eager is None:
        enforce_eager = gpu_capabilities["should_use_eager_attention"]
        logger.info(
            f"Auto-configuring eager mode: {enforce_eager} based on GPU capabilities"
        )

    # Initialize model_kwargs with basic values
    model_kwargs = {
        "model": model_id,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "seed": seed,
        "dtype": dtype,
    }

    # Add optional parameters only if specified
    if max_model_len is not None:
        model_kwargs["max_model_len"] = max_model_len

    if cpu_offload_gb > 0:
        model_kwargs["cpu_offload"] = True
        model_kwargs["offload_offsets"] = True
        logger.info(f"Enabling CPU offloading with {cpu_offload_gb}GB threshold")

    # Add enforce_eager if it's True
    if enforce_eager:
        model_kwargs["enforce_eager"] = True
        logger.info("Enforcing eager mode (CUDAGraph disabled)")

    # Load the dataset
    logger.info("Loading GSM8K dataset")
    try:
        gsm8k = load_gsm8k("test")
        logger.info(f"Loaded {len(gsm8k)} examples from GSM8K test set")
    except Exception as e:
        logger.error(f"Failed to load GSM8K dataset: {e}")
        return {
            "error": f"Dataset loading failed: {str(e)}",
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }

    # Select examples for benchmark
    np.random.seed(seed)
    try:
        if num_samples > 0 and num_samples < len(gsm8k):
            selected_indices = np.random.choice(len(gsm8k), num_samples, replace=False)
            benchmark_examples = [gsm8k[int(i)] for i in selected_indices]
            remaining_indices = list(set(range(len(gsm8k))) - set(selected_indices))
            shot_indices = np.random.choice(
                remaining_indices, num_few_shot, replace=False
            )
        else:
            benchmark_examples = [gsm8k[i] for i in range(len(gsm8k))]
            shot_indices = np.random.choice(len(gsm8k), num_few_shot, replace=False)

        few_shot_examples = [gsm8k[int(i)] for i in shot_indices]
        logger.info(
            f"Selected {len(benchmark_examples)} examples for benchmark and {num_few_shot} examples for few-shot prompting"
        )
    except Exception as e:
        logger.error(f"Failed to select examples: {e}")
        return {
            "error": f"Example selection failed: {str(e)}",
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
        }

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model {model_id} using {dtype} precision")
    start_time = time.time()

    try:
        # Log the final model_kwargs for debugging
        logger.info(
            f"Model configuration: {json.dumps({k: str(v) for k, v in model_kwargs.items()}, indent=2)}"
        )

        # Initialize the model
        model = LLM(**model_kwargs)

        model_load_time = time.time() - start_time
        logger.info(f"Model loaded in {model_load_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Try with different configurations if initial loading fails
        if "enforce_eager" not in model_kwargs and not enforce_eager:
            logger.info("Trying with enforce_eager=True")
            return run_benchmark(
                model_id=model_id,
                system_prompt=system_prompt,
                num_samples=num_samples,
                num_few_shot=num_few_shot,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                log_dir=log_dir,
                output_dir=output_dir,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                cpu_offload_gb=cpu_offload_gb,
                dtype=dtype,
                enforce_eager=True,
                batch_size=batch_size,
            )
        # If BF16 fails, try FP16
        elif dtype == "bfloat16":
            logger.info("BFloat16 failed, trying with Float16")
            return run_benchmark(
                model_id=model_id,
                system_prompt=system_prompt,
                num_samples=num_samples,
                num_few_shot=num_few_shot,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                log_dir=log_dir,
                output_dir=output_dir,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                cpu_offload_gb=cpu_offload_gb,
                dtype="float16",
                enforce_eager=enforce_eager,
                batch_size=batch_size,
            )
        return {
            "error": str(e),
            "run_id": run_id,
            "model_id": model_id,
            "model_load_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "system_prompt": system_prompt,
                "num_few_shot": num_few_shot,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "dtype": dtype,
                "enforce_eager": enforce_eager,
                "batch_size": batch_size,
            },
        }

    # Format few-shot examples using appropriate chat template for the model
    try:
        few_shot_base = format_five_shot_multiturn(
            few_shot_examples, system_prompt, model
        )
        logger.info("Formatted 5-shot multi-turn examples")
    except Exception as e:
        logger.error(f"Failed to format few-shot examples: {e}")
        logger.info("Using fallback formatting")
        # Simple fallback formatting
        few_shot_base = system_prompt + "\n\n"
        for example in few_shot_examples:
            few_shot_base += (
                f"User: {example['question']}\nAssistant: {example['answer']}\n\n"
            )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Results storage with enhanced granularity and standard GSM8K metrics
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
            "max_model_len": max_model_len,
        },
        "hardware_info": check_gpu_capabilities(),
        "timing": {
            "start_time": datetime.now().isoformat(),
            "model_load_time": model_load_time,
        },
        "examples": [],
        "metrics": {
            "answer_accuracy": {
                "exact_match": 0.0,  # Exact numerical match
                "approximate_match": 0.0,  # Within small epsilon
                "total_correct": 0,
                "total_examples": 0,
            },
            "reasoning": {
                "valid_solution_path": 0.0,  # Percentage with valid reasoning steps
                "step_accuracy": 0.0,  # Accuracy of intermediate steps
                "avg_num_steps": 0.0,  # Average number of reasoning steps
                "step_by_step_present": 0.0,  # Percentage using explicit steps
                "math_format_accuracy": 0.0,  # Correct mathematical notation usage
            },
            "complexity_analysis": {
                "single_step_accuracy": 0.0,  # Accuracy on single-step problems
                "multi_step_accuracy": 0.0,  # Accuracy on multi-step problems
                "avg_steps_per_solution": 0.0,
                "max_steps_solved": 0,  # Maximum number of steps in correctly solved problems
            },
            "error_analysis": {
                "calculation_errors": 0.0,  # Percentage with calculation mistakes
                "reasoning_errors": 0.0,  # Percentage with logical mistakes
                "incomplete_solutions": 0.0,  # Percentage without final answer
                "invalid_steps": 0.0,  # Percentage with invalid intermediate steps
            },
            "avg_nlg_metrics": {
                "bleu": 0.0,
                "meteor": 0.0,
                "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
                "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            },
        },
        "performance": {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_inference_time": 0,
            "tokens_per_second": 0,
        },
    }

    try:
        # Create progress bar for overall progress
        from tqdm import tqdm

        total_examples = len(benchmark_examples)
        progress_bar = tqdm(
            total=total_examples,
            desc="GSM8K Progress",
            leave=True,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
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
                available_examples = [
                    ex
                    for i, ex in enumerate(benchmark_examples)
                    if i < batch_idx or i >= batch_end
                ]
                few_shot_examples = np.random.choice(
                    available_examples,
                    min(num_few_shot, len(available_examples)),
                    replace=False,
                ).tolist()
            else:
                few_shot_examples = []

            # Format few-shot examples
            few_shot_base = format_five_shot_multiturn(
                few_shot_examples, system_prompt, model
            )

            # Prepare prompts for the batch
            full_prompts = []
            for example in current_batch:
                if hasattr(model, "tokenizer") and hasattr(
                    model.tokenizer, "apply_chat_template"
                ):
                    conversation = [{"role": "system", "content": system_prompt}]

                    # Add few-shot examples
                    for fs_example in few_shot_examples:
                        conversation.append(
                            {"role": "user", "content": fs_example["question"]}
                        )
                        conversation.append(
                            {"role": "assistant", "content": fs_example["answer"]}
                        )

                    # Add current question
                    conversation.append(
                        {"role": "user", "content": example["question"]}
                    )

                    prompt = model.tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = few_shot_base + f"User: {example['question']}\nAssistant:"

                full_prompts.append(prompt)

            # Get model responses
            batch_start_time = time.time()
            outputs = model.generate(
                prompts=full_prompts, sampling_params=sampling_params
            )
            batch_inference_time = time.time() - batch_start_time

            # Process results
            batch_correct = 0
            for idx in range(batch_size_actual):
                example = current_batch[idx]
                output = outputs[idx]

                # Get generated text and extract answer
                generated_text = output.outputs[0].text
                predicted_answer = extract_answer(generated_text)
                ground_truth_answer = extract_ground_truth(example["answer"])

                # Calculate correctness
                is_correct = False
                is_approx_correct = False
                if predicted_answer is not None and ground_truth_answer is not None:
                    is_exact_match = abs(predicted_answer - ground_truth_answer) < 1e-6
                    is_approx_match = (
                        abs(predicted_answer - ground_truth_answer) < 0.05
                    )  # 5% tolerance
                    is_correct = is_exact_match

                    if is_exact_match:
                        results["metrics"]["answer_accuracy"]["total_correct"] += 1
                        results["metrics"]["answer_accuracy"]["exact_match"] += 1
                        batch_correct += 1
                    elif is_approx_match:
                        results["metrics"]["answer_accuracy"]["approximate_match"] += 1

                # Calculate NLG metrics
                nlg_metrics = calculate_nlg_metrics(generated_text, example["answer"])

                # Store detailed example results
                example_result = {
                    "example_id": batch_idx + idx,
                    "question": example["question"],
                    "ground_truth": example["answer"],
                    "ground_truth_answer": ground_truth_answer,
                    "few_shot_examples": [
                        {"question": fs_ex["question"], "answer": fs_ex["answer"]}
                        for fs_ex in few_shot_examples
                    ],
                    "full_prompt": full_prompts[idx],
                    "model_output": generated_text,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "metrics": {
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(output.outputs[0].token_ids),
                        "total_tokens": len(output.prompt_token_ids)
                        + len(output.outputs[0].token_ids),
                        "inference_time": batch_inference_time / batch_size_actual,
                        "tokens_per_second": (
                            len(output.prompt_token_ids)
                            + len(output.outputs[0].token_ids)
                        )
                        / (batch_inference_time / batch_size_actual),
                    },
                    "nlg_metrics": nlg_metrics,
                    "batch_info": {
                        "batch_size": batch_size_actual,
                        "batch_inference_time": batch_inference_time,
                        "batch_index": batch_idx // batch_size,
                    },
                    "solution_analysis": {
                        "step_count": len(generated_text.split("\n")),
                        "answer_position": len(generated_text)
                        - len(generated_text.split("####")[-1])
                        if "####" in generated_text
                        else len(generated_text),
                        "has_step_markers": "####" in generated_text
                        or "Let's solve this step by step" in generated_text,
                    },
                    "timestamp": datetime.now().isoformat(),
                }

                results["examples"].append(example_result)
                results["metrics"]["answer_accuracy"]["total_examples"] += 1

                # Update performance metrics
                results["performance"]["total_tokens"] += example_result["metrics"][
                    "total_tokens"
                ]
                results["performance"]["total_prompt_tokens"] += example_result[
                    "metrics"
                ]["prompt_tokens"]
                results["performance"]["total_completion_tokens"] += example_result[
                    "metrics"
                ]["completion_tokens"]
                results["performance"]["total_inference_time"] += example_result[
                    "metrics"
                ]["inference_time"]

                # Update NLG metrics
                for metric, value in nlg_metrics.items():
                    try:
                        if metric.startswith("rouge"):
                            metric_name, score_type = metric.split("_")
                            if (
                                metric_name in results["metrics"]["avg_nlg_metrics"]
                                and score_type
                                in results["metrics"]["avg_nlg_metrics"][metric_name]
                            ):
                                results["metrics"]["avg_nlg_metrics"][metric_name][
                                    score_type
                                ] += value
                        elif metric in results["metrics"][
                            "avg_nlg_metrics"
                        ] and not metric.startswith("error"):
                            results["metrics"]["avg_nlg_metrics"][metric] += value
                    except (KeyError, ValueError, TypeError) as e:
                        # Silently ignore problematic metrics to avoid crashing the entire benchmark
                        pass

            # Update progress and accuracy
            total_processed += batch_size_actual
            total_correct += batch_correct
            current_accuracy = (
                total_correct / total_processed if total_processed > 0 else 0.0
            )

            progress_bar.update(batch_size_actual)
            progress_bar.set_postfix({"Acc": f"{current_accuracy:.2%}"})

        progress_bar.close()

        # Add end time and total time
        end_time = datetime.now()
        results["timing"]["end_time"] = end_time.isoformat()
        results["timing"]["total_run_time"] = (
            end_time - datetime.fromisoformat(results["timing"]["start_time"])
        ).total_seconds()

        # Calculate final metrics
        if results["metrics"]["answer_accuracy"]["total_examples"] > 0:
            total_examples = results["metrics"]["answer_accuracy"]["total_examples"]

            # Calculate accuracy percentages
            results["metrics"]["answer_accuracy"]["accuracy"] = (
                results["metrics"]["answer_accuracy"]["total_correct"] / total_examples
            )
            results["metrics"]["answer_accuracy"]["exact_match"] = (
                results["metrics"]["answer_accuracy"]["exact_match"] / total_examples
            )
            results["metrics"]["answer_accuracy"]["approximate_match"] = (
                results["metrics"]["answer_accuracy"]["approximate_match"]
                / total_examples
            )

            # Calculate reasoning metrics
            step_by_step_count = sum(
                1
                for ex in results["examples"]
                if ex["solution_analysis"]["has_step_markers"]
            )
            results["metrics"]["reasoning"]["step_by_step_present"] = (
                step_by_step_count / total_examples
            )

            # Calculate average steps per solution
            avg_steps = (
                sum(ex["solution_analysis"]["step_count"] for ex in results["examples"])
                / total_examples
            )
            results["metrics"]["reasoning"]["avg_num_steps"] = avg_steps
            results["metrics"]["complexity_analysis"]["avg_steps_per_solution"] = (
                avg_steps
            )

            # Performance metrics
            results["performance"]["tokens_per_second"] = (
                results["performance"]["total_tokens"]
                / results["performance"]["total_inference_time"]
            )

            # Average NLG metrics
            for metric in results["metrics"]["avg_nlg_metrics"]:
                if isinstance(results["metrics"]["avg_nlg_metrics"][metric], dict):
                    for score_type in results["metrics"]["avg_nlg_metrics"][metric]:
                        results["metrics"]["avg_nlg_metrics"][metric][score_type] /= (
                            total_examples
                        )
                else:
                    results["metrics"]["avg_nlg_metrics"][metric] /= total_examples

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        # Save detailed results
        detailed_results_file = os.path.join(
            output_dir, f"detailed_results_{run_id}.json"
        )
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
            "performance": results["performance"],
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
            logger.info(
                f"Saving partial results from {len(results['examples'])} examples"
            )
            return results
        else:
            return {
                "error": str(e),
                "run_id": run_id,
                "model_id": model_id,
                "model_load_time": model_load_time,
                "timestamp": datetime.now().isoformat(),
                "run_config": results["run_config"],
            }


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM on GSM8K with few-shot prompting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required arguments
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model ID or local path",
    )

    # Optional model configuration arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant that solves math problems step by step.",
        help="System prompt for the model",
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    model_group.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    model_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length for the model",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32", "auto"],
        default="auto",
        help="Datatype to use for model weights",
    )
    model_group.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp16", "bf16"],
        default="auto",
        help="KV cache data type",
    )

    # Hardware utilization arguments
    hw_group = parser.add_argument_group("Hardware Utilization")
    hw_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0-1.0)",
    )
    hw_group.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    hw_group.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0,
        help="Amount of GPU memory to offload to CPU (in GB)",
    )
    hw_group.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enforce eager mode (disable CUDA graph)",
    )
    hw_group.add_argument(
        "--disable-cuda-graphs",
        action="store_true",
        help="Explicitly disable CUDA graphs",
    )
    hw_group.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Maximum number of concurrent sequences",
    )
    hw_group.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=None,
        help="Chunk size for prefill phase (if supported)",
    )
    hw_group.add_argument(
        "--disable-bestpath",
        action="store_true",
        help="Disable bestpath scheduling optimization",
    )

    # Benchmark configuration arguments
    bench_group = parser.add_argument_group("Benchmark Configuration")
    bench_group.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to benchmark (0 for all)",
    )
    bench_group.add_argument(
        "--num-few-shot", type=int, default=5, help="Number of few-shot examples to use"
    )
    bench_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    bench_group.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing examples"
    )

    # Output configuration arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--log-dir", type=str, default="logs", help="Directory to save logs"
    )
    output_group.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )
    output_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode with additional logging"
    )
    output_group.add_argument(
        "--quiet", action="store_true", help="Reduce verbosity of output"
    )
    output_group.add_argument(
        "--nltk-data-dir",
        type=str,
        default=None,
        help="Custom directory to store NLTK data",
    )

    args = parser.parse_args()

    # Set environment variables based on args
    if args.debug:
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    if args.quiet:
        os.environ["VLLM_DISABLE_TQDM"] = "1"
        os.environ["VLLM_LOG_LEVEL"] = "ERROR"
        os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # Setup NLTK resources if specified
    if args.nltk_data_dir:
        print(f"Using custom NLTK data directory: {args.nltk_data_dir}")
        setup_nltk(args.nltk_data_dir)

    try:
        # Verify NLTK resources
        nltk = get_nltk()
        nltk.word_tokenize("Testing NLTK initialization.")
        print("NLTK resources verified successfully.")
    except Exception as e:
        print(f"NLTK resource issue detected: {e}")
        print("Attempting to download missing NLTK resources...")
        setup_nltk(args.nltk_data_dir)

    # Handle dtype and other parameter configuration
    dtype = None if args.dtype == "auto" else args.dtype
    kv_cache_dtype = None if args.kv_cache_dtype == "auto" else args.kv_cache_dtype
    enforce_eager = args.enforce_eager or args.disable_cuda_graphs

    # Run the benchmark
    start_time = time.time()
    print(f"Starting benchmark for model: {args.model_id}")

    try:
        results = run_benchmark(
            model_id=args.model_id,
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

    # Print accuracy metrics if available
    if "metrics" in results and "answer_accuracy" in results["metrics"]:
        acc_metrics = results["metrics"]["answer_accuracy"]
        print(f"Total examples: {acc_metrics.get('total_examples', 0)}")
        print(f"Exact match accuracy: {acc_metrics.get('exact_match', 0.0):.4f}")
        print(
            f"Approximate match accuracy: {acc_metrics.get('approximate_match', 0.0):.4f}"
        )
        print(f"Overall accuracy: {acc_metrics.get('accuracy', 0.0):.4f}")
    else:
        print(f"Total examples: {results.get('total_examples', 0)}")
        print(f"Overall accuracy: {results.get('accuracy', 0.0):.4f}")

    # Print performance metrics if available
    if "performance" in results:
        perf = results["performance"]
        print(f"Total tokens: {perf.get('total_tokens', 0)}")
        print(f"Tokens per second: {perf.get('tokens_per_second', 0.0):.2f}")
        print(f"Total inference time: {perf.get('total_inference_time', 0.0):.2f}s")

    print(f"Total benchmark time: {total_time:.2f}s")

    # Print error if any
    if "error" in results:
        print(f"\nERROR during benchmark: {results['error']}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
