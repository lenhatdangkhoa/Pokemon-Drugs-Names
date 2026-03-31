"""
VLLM setup and validation for Pokemon experiment.
"""

import logging
import os
from typing import Optional, Tuple

# Try to import VLLM
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    StructuredOutputsParams = None  # type: ignore[misc, assignment]

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False

from src.utils import get_available_gpu_count


def setup_vllm_mode(args) -> Tuple:
    """Setup VLLM mode for Pokemon experiment."""
    if not VLLM_AVAILABLE:
        raise ImportError("VLLM package not available. Install with: pip install vllm")
    if not TOKENIZER_AVAILABLE:
        raise ImportError("transformers package not available. Install with: pip install transformers")

    # Setup model with memory optimization parameters
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=args.number_gpus,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,  # Reduce GPU memory utilization to avoid OOM
        max_num_seqs=128,           
         # Reduce max concurrent sequences to save memory
        enforce_eager=True,          # Disable CUDA graphs to reduce memory usage
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Setup sampling parameters for main model
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    # Judge: constrain decoding to exactly "[0]", "[1]", or "[2]" so greedy decoding cannot
    # collapse into degenerate punctuation (e.g. repeated "!"), which breaks parsing.
    if StructuredOutputsParams is not None:
        judge_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=16,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            structured_outputs=StructuredOutputsParams(
                choice=["[0]", "[1]", "[2]"],
            ),
        )
    else:
        judge_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=32,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.15,
            stop=["\n"],
        )

    # Setup LoRA if specified
    lora_request = LoRARequest(
        lora_name="lora_adapter",
        lora_int_id=1,
        lora_path=args.lora_path,
    ) if args.lora_path else None

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    return model, tokenizer, sampling_params, judge_sampling_params, lora_request


def validate_vllm_args(args) -> None:
    """Validate VLLM-specific arguments."""
    if args.mode == "vllm":
        if not VLLM_AVAILABLE:
            raise ValueError("VLLM mode selected but VLLM package not available")

        if args.number_gpus <= 0:
            raise ValueError(f"number_gpus must be > 0 for VLLM mode, got {args.number_gpus}")

        available_gpus = get_available_gpu_count()
        if args.number_gpus > available_gpus:
            raise ValueError(f"Requested {args.number_gpus} GPUs but only {available_gpus} available")

        # vLLM tensor parallel requires num_attention_heads % tensor_parallel_size == 0
        if TOKENIZER_AVAILABLE and args.number_gpus > 1:
            try:
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(
                    args.model_name, trust_remote_code=True
                )
            except Exception as e:
                logging.warning(
                    "Could not load model config to validate tensor parallel vs heads: %s",
                    e,
                )
            else:
                n_heads = getattr(cfg, "num_attention_heads", None)
                if n_heads and n_heads % args.number_gpus != 0:
                    valid = [
                        d
                        for d in range(1, min(available_gpus, n_heads) + 1)
                        if n_heads % d == 0
                    ]
                    raise ValueError(
                        f"number_gpus={args.number_gpus} is invalid for {args.model_name}: "
                        f"tensor parallel size must divide num_attention_heads={n_heads}. "
                        f"Try one of: {valid}"
                    )


def validate_openai_args(args) -> None:
    """Validate OpenAI-specific arguments."""
    import os
    from src.client_factory import OPENAI_AVAILABLE
    
    if args.mode == "openai":
        # Check if model name starts with "azure-" - if so, should use Azure mode instead
        if args.model_name.startswith("azure-"):
            raise ValueError(
                f"Model name '{args.model_name}' starts with 'azure-' but mode is 'openai'. "
                "Either use --mode azure or remove 'azure-' prefix from model name."
            )
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI mode selected but OpenAI package not available")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")


def validate_azure_args(args) -> None:
    """Validate Azure OpenAI-specific arguments."""
    import os
    from src.client_factory import AZURE_AVAILABLE, AZURE_IDENTITY_AVAILABLE
    
    if args.mode == "azure":
        if not AZURE_AVAILABLE or not AZURE_IDENTITY_AVAILABLE:
            raise ValueError("Azure OpenAI mode selected but Azure packages not available. Install with: pip install azure-identity openai")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            logging.warning("AZURE_OPENAI_ENDPOINT not found in environment, using default endpoint")
