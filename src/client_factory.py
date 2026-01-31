"""
Client factory for OpenAI and Azure OpenAI clients.

Provides client initialization and caching functionality.
"""

import logging
import os
import time
from typing import Dict, List, Tuple

# Conditional imports
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
    AZURE_AVAILABLE = True
except ImportError:
    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        AZURE_AVAILABLE = False
    except ImportError:
        OPENAI_AVAILABLE = False
        AZURE_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

# Client cache to avoid reinitializing clients for each API call
_client_cache: Dict[str, tuple] = {}

# Pricing for Azure OpenAI models (per 1K tokens)
# GPT-5-chat pricing (estimated based on Azure OpenAI pricing structure)
AZURE_PRICING = {
    "gpt-5-chat": {
        "input": 0.01,   # $0.01 per 1K input tokens
        "output": 0.03,  # $0.03 per 1K output tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015,  # $0.00015 per 1K input tokens
        "output": 0.0006,  # $0.0006 per 1K output tokens
    },
    # Default pricing if model not found
    "default": {
        "input": 0.01,
        "output": 0.03,
    }
}

# Cost tracker class
class CostTracker:
    """Track token usage and costs for API calls."""
    
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.model_name = None
    
    def add_usage(self, usage_data: Dict):
        """Add usage data from an API response."""
        if usage_data:
            self.total_input_tokens += usage_data.get("prompt_tokens", 0)
            self.total_output_tokens += usage_data.get("completion_tokens", 0)
            self.total_requests += 1
            if not self.model_name and usage_data.get("model"):
                self.model_name = usage_data.get("model")
    
    def get_cost(self) -> float:
        """Calculate total cost based on model pricing."""
        if not self.model_name:
            return 0.0
        
        # Get base model name (remove azure- prefix if present)
        base_model = self.model_name.replace("azure-", "")
        
        # Get pricing for this model
        pricing = AZURE_PRICING.get(base_model, AZURE_PRICING["default"])
        
        input_cost = (self.total_input_tokens / 1000.0) * pricing["input"]
        output_cost = (self.total_output_tokens / 1000.0) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict:
        """Get summary of usage and costs."""
        return {
            "model": self.model_name or "unknown",
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": self.get_cost(),
        }

# Global cost tracker instance
_global_cost_tracker = CostTracker()


def _initialize_client(model_name: str) -> Tuple:
    """
    Initialize OpenAI or Azure OpenAI client based on model name.
    Uses caching to avoid reinitializing clients for repeated calls.
    
    Args:
        model_name: Model name (e.g., "gpt-4o-mini" or "azure-gpt-4o-mini")
        
    Returns:
        Tuple of (client, model_param) where model_param is the deployment/model name to use
    """
    # Return cached client if available
    if model_name in _client_cache:
        return _client_cache[model_name]
    
    if model_name.startswith("azure-"):
        # Azure OpenAI configuration (no fallback - only Entra ID authentication)
        if not AZURE_AVAILABLE or not AZURE_IDENTITY_AVAILABLE:
            logging.warning(
                "⚠️  Azure OpenAI or Azure Identity not available. "
                "Please install: pip install azure-identity openai"
            )
            raise ValueError(
                "Azure OpenAI or Azure Identity not available. "
                "Please install: pip install azure-identity openai"
            )
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai-endpoint.openai.azure.com/")
        deployment = model_name.replace("azure-", "")  # Remove azure- prefix
        
        # Initialize Azure OpenAI client with Entra ID authentication only
        # Exclude Managed Identity to avoid delays on local machines (macOS)
        credential = DefaultAzureCredential(
            exclude_managed_identity_credential=True,
        )
        token_provider = get_bearer_token_provider(
            credential,
            "https://your-cognitive-services-endpoint/.default"
        )
        
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version="2024-02-15-preview",
        )
        logging.info(f"✅ Using Azure OpenAI: {deployment} at {endpoint}")
        
        model_param = deployment
    else:
        # Standard OpenAI configuration (no fallback)
        if not OPENAI_AVAILABLE:
            logging.warning("⚠️  OpenAI package not available. Install with: pip install openai")
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.warning("⚠️  OPENAI_API_KEY not found in environment.")
            raise EnvironmentError("OPENAI_API_KEY not found in environment.")
        
        client = OpenAI(api_key=api_key)
        model_param = model_name
        logging.info(f"✅ Using OpenAI: {model_name}")
    
    # Cache the client for reuse
    _client_cache[model_name] = (client, model_param)
    return client, model_param


def get_completion_from_messages(
    messages: List[Dict], 
    model: str = "gpt-4o-mini",
    mode: str = "openai",
    temperature: float = 0.7, 
    max_tokens: int = 2048,
    top_p: float = 0.9,
    max_retries: int = 5,
    initial_backoff: float = 1.0
) -> Tuple[str, Dict]:
    """
    Get completion from OpenAI or Azure OpenAI API with exponential backoff retry logic.
    
    Args:
        messages: List of message dictionaries
        model: Model name (e.g., "gpt-4o-mini", "gpt-5-chat")
        mode: Mode ("openai" or "azure")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        max_retries: Maximum number of retry attempts for rate limit errors
        initial_backoff: Initial backoff time in seconds (doubles with each retry)
        
    Returns:
        Generated response text
    """
    # If mode is azure, prefix model name with "azure-" if not already present
    if mode == "azure" and not model.startswith("azure-"):
        model = f"azure-{model}"
    client, model_param = _initialize_client(model)
    
    backoff_time = initial_backoff
    response = None
    
    for attempt in range(max_retries + 1):
        try:
            logging.debug(f"Making API call (attempt {attempt + 1}/{max_retries + 1}): model={model_param}, temp={temperature}, max_tokens={max_tokens}")

            response = client.chat.completions.create(
                model=model_param,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            break  # Success, exit retry loop

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate limit" in error_str or "too many requests" in error_str
            
            if is_rate_limit and attempt < max_retries:
                logging.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                logging.info(f"Waiting {backoff_time:.1f} seconds before retry...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                continue
            else:
                # Not a rate limit error, or max retries reached
                if attempt >= max_retries:
                    logging.error(f"Max retries ({max_retries}) reached. Last error: {e}")
                raise
    
    if response is None:
        raise RuntimeError("Failed to get response after all retry attempts")
    
    # Extract usage information from response
    usage_data = {}
    if hasattr(response, 'usage'):
        usage = response.usage
        usage_data = {
            "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage, 'completion_tokens', 0),
            "total_tokens": getattr(usage, 'total_tokens', 0),
            "model": model_param,
        }
        # Track usage globally
        _global_cost_tracker.add_usage(usage_data)
    
    # Handle different response formats
    try:
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content.strip()
                return content, usage_data
            elif hasattr(choice, 'text'):  # Some APIs return 'text' instead of 'message.content'
                content = choice.text.strip()
                return content, usage_data
            else:
                logging.error(f"Unexpected choice structure: {choice}")
                raise ValueError(f"Cannot extract content from response choice: {choice}")
        else:
            logging.error(f"No choices in response: {response}")
            raise ValueError("No choices in API response")
    except Exception as e:
        logging.error(f"Error parsing API response: {e}")
        logging.error(f"Response structure: {response}")
        raise


def get_cost_summary() -> Dict:
    """Get cost summary from global tracker."""
    return _global_cost_tracker.get_summary()


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_cost_tracker
    _global_cost_tracker = CostTracker()
