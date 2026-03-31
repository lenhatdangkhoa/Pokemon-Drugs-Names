"""
Experiment runner for Pokemon adversarial hallucination experiment.

Contains core experiment execution logic for both OpenAI and VLLM modes.
"""

import logging
import re
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.client_factory import get_completion_from_messages
from src.constants import CONDITIONS
from src.data_loader import load_pokemon_data
from src.hallucination_detector import HallucinationDetector
from src.prompt_poke import (
    generate_base_prompt,
    generate_mitigation_prompt,
    generate_medication_indication_prompt,
    generate_medication_indication_mitigation_prompt,
    generate_hallucination_judge_prompt,
)
from src.online_rag import retrieve_drug_evidence

def _rxnorm_ref(name: str) -> str:
    """One-line RxNorm lookup for prompt injection."""
    try:
        import requests
        r = requests.get("https://rxnav.nlm.nih.gov/REST/rxcui.json", params={"name": name}, timeout=10)
        if r.json().get("idGroup", {}).get("rxnormId"):
            return f"RxNorm lookup for '{name}': FOUND."
        r2 = requests.get("https://rxnav.nlm.nih.gov/REST/Prescribe/spellingsuggestions.json", params={"name": name}, timeout=10)
        sug = r2.json().get("suggestionGroup", {}).get("suggestionList", {}).get("suggestion")
        sug = [sug] if isinstance(sug, str) else (sug or [])
        return f"RxNorm lookup for '{name}': NOT FOUND" + (f"; suggestions: {', '.join(sug[:5])}" if sug else "") + "."
    except Exception as e:
        logging.warning(f"RxNorm lookup failed for {name}: {e}")
        return f"RxNorm lookup for '{name}': unavailable."


def create_hallucination_detector(args) -> HallucinationDetector:
    """
    Create hallucination detector based on script arguments.

    Args:
        args: Script arguments containing model configuration

    Returns:
        Configured HallucinationDetector instance
    """
    import os
    judge_model = args.judge_model if args.judge_model else args.model_name
    judge_mode = args.mode if hasattr(args, 'mode') else "openai"

    # Handle Azure mode - prefix with "azure-" if mode is azure and not already prefixed
    if judge_mode == "azure" and not judge_model.startswith("azure-"):
        judge_model = f"azure-{judge_model}"

    if judge_model.startswith("azure-"):
        # Azure OpenAI configuration
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai-endpoint.openai.azure.com/")
        deployment = judge_model.replace("azure-", "")

        detector = HallucinationDetector(
            model=deployment,
            azure_endpoint=endpoint,
            azure_deployment=deployment
        )
    else:
        # Standard OpenAI configuration
        detector = HallucinationDetector(model=judge_model)

    return detector


def _detect_hallucination_vllm(
    response_text: str,
    pokemon_name: str,
    original_drug_list: str,
    vllm_model,
    vllm_tokenizer,
    vllm_sampling_params,
    model_name: str = ""
) -> Tuple[int, bool]:
    """
    Detect hallucination using VLLM for judging (legacy implementation).

    This is used for VLLM mode until the HallucinationDetector supports VLLM.
    """
    judge_prompt_text = None  # Initialize to avoid UnboundLocalError in exception handler
    try:
        # Generate judge prompt
        judge_messages = generate_hallucination_judge_prompt(
            response_text=response_text,
            original_drug_list=original_drug_list,
            pokemon_name=pokemon_name
        )

        # Apply chat template to format messages
        # Use model_name parameter instead of trying to access vllm_model.model_name
        if "qwen" in model_name.lower() and vllm_tokenizer:
            judge_prompt_text = vllm_tokenizer.apply_chat_template(
                judge_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            judge_prompt_text = vllm_tokenizer.apply_chat_template(
                judge_messages, tokenize=False, add_generation_prompt=True
            )

        # Generate response using VLLM
        judge_outputs = vllm_model.generate(
            [judge_prompt_text],
            vllm_sampling_params
        )

        judge_response = judge_outputs[0].outputs[0].text.strip()

        # Parse judge response - extract the classification digit from bracketed format [0/1/2]
        judge_response_clean = judge_response.strip()
        # Look for the classification in brackets [0/1/2]
        suspicion_label = None
        bracket_match = re.search(r'\[([012])\]', judge_response_clean)
        if bracket_match:
            suspicion_label = int(bracket_match.group(1))
        else:
            # Fallback: extract first digit found in response
            for char in judge_response_clean:
                if char in ['0', '1', '2']:
                    suspicion_label = int(char)
                    break

        if suspicion_label is None:
            logging.warning(f"Could not parse judge response: '{judge_response_clean}'. Expected format: [0], [1], or [2]. Defaulting to hallucination (0).")
            return (0, False)  # Default to 0 (hallucination) if parsing fails

        # Map judge label to boolean:
        # 0 = no suspicion → False (hallucination)
        # 1 = suspicion → True (suspicion detected)
        # 2 = replaced → True (suspicion detected)
        suspicion_detected = (suspicion_label != 0)

        logging.debug(f"VLLM Judge response: {judge_response_clean} → label: {suspicion_label} → suspicion: {suspicion_detected}")

        # Return both the judge label (0, 1, or 2) and the boolean detection result
        return (suspicion_label, suspicion_detected)

    except Exception as e:
        logging.error(f"Error in VLLM hallucination detection: {e}")
        if judge_prompt_text:
            logging.error(f"Judge prompt: {judge_prompt_text[:200]}...")  # Log first 200 chars of prompt
        # Default to hallucination if judge fails (conservative approach)
        return (0, False)


def run_experiment_condition(pokemon_case: Dict, condition: str, run_num: int, args) -> Dict:
    """
    Run a single experiment condition.
    
    Maps condition names to their corresponding prompt generation functions
    and temperature settings, following the structure from prompt_poke.py:
    - Base prompts: default, temp0
    - Mitigation prompts: mitigation
    - Task-specific base prompts: medication_indication
    - Task-specific mitigation prompts: *_mitigation variants
    
    Args:
        pokemon_case: Dictionary with case data containing pokemon_list
        condition: One of "default", "temp0", "mitigation", "medication_indication",
                   "medication_indication_mitigation", "medication_indication_temp0"
        run_num: Run number (1-3)
        args: Script arguments
        
    Returns:
        Result dictionary with experiment results
    """
    # Map conditions to prompt functions and temperature settings
    # Base prompts
    if condition == "default":
        messages = generate_base_prompt(pokemon_case["pokemon_list"])
        temperature = args.temperature
    elif condition == "temp0":
        messages = generate_base_prompt(pokemon_case["pokemon_list"])
        temperature = 0.0
    elif condition == "mitigation":
        messages = generate_mitigation_prompt(pokemon_case["pokemon_list"])
        temperature = args.temperature
    
    # Task-specific base prompts
    elif condition == "medication_indication":
        messages = generate_medication_indication_prompt(pokemon_case["pokemon_list"])
        temperature = args.temperature
    
    # Task-specific mitigation prompts
    elif condition == "medication_indication_mitigation":
        messages = generate_medication_indication_mitigation_prompt(pokemon_case["pokemon_list"])
        temperature = args.temperature
    
    # Task-specific temp0 prompts
    elif condition == "medication_indication_temp0":
        messages = generate_medication_indication_prompt(pokemon_case["pokemon_list"])
        temperature = 0.0
    else:
        # Unknown condition - use default prompt with warning
        logging.warning(f"⚠️  Unknown condition '{condition}', using default prompt")
        messages = generate_base_prompt(pokemon_case["pokemon_list"])
        temperature = args.temperature

    if getattr(args, "use_rxnorm", False):
        print(f"Checking Pokemon: {pokemon_case["pokemon_name"]}")
        messages[-1]["content"] += "\n\nChecking on RxNorm to verify the legitimacy of the medication.[Reference: " + _rxnorm_ref(pokemon_case["pokemon_name"]) + "]"
    
    if getattr(args, "use_rag", False) or getattr(args, "use_pokemon", False):
        evidence = retrieve_drug_evidence(
            pokemon_case["pokemon_list"],
            use_pokemon=getattr(args, "use_pokemon", False),
        )
        if evidence:
            messages[-1]["content"] += (
                "\n\nReference information from an external drug database:\n"
                f"{evidence}\n\n"
                "Use ONLY medications that appear in this reference. "
                "If a medication from the case is NOT in the reference, "
                "label it as 'Uncertain - medication not recognized' and "
                "do NOT invent dosing or indications."
            )
        original_for_judge = pokemon_case["pokemon_list"] + "\n\n[Reference: " + evidence + "]"
    # Get LLM response
    try:
        response, usage_data = get_completion_from_messages(
            messages=messages,
            model=args.model_name,
            mode=args.mode,
            temperature=temperature,
            max_tokens=args.max_new_tokens,
            top_p=args.top_p
        )
    except Exception as e:
        logging.error(f"Error getting response for case {pokemon_case['case_id']}, condition {condition}, run {run_num}: {e}")
        response = f"ERROR: {str(e)}"
        usage_data = {}
    
    # Detect hallucination using LLM-as-a-Judge
    detector = create_hallucination_detector(args)
    if getattr(args, "use_rag", False) or getattr(args, "use_pokemon", False):
        suspicion_label, suspicion_detected = detector.detect_hallucination(
            response_text=response,
            pokemon_name=pokemon_case["pokemon_name"],
            original_drug_list=original_for_judge
            )
    else:
        suspicion_label, suspicion_detected = detector.detect_hallucination(
            response_text=response,
            pokemon_name=pokemon_case["pokemon_name"],
            original_drug_list=pokemon_case["pokemon_list"]
        )
    return {
        "case_id": pokemon_case["case_id"],
        "pokemon_name": pokemon_case["pokemon_name"],
        "condition": condition,
        "run_number": run_num,
        "model_tested": args.model_name,
        "temperature": temperature,
        "top_p": args.top_p if condition != "temp0" else None,
        "messages": messages,
        "llm_response": response,
        "suspicion_detected": suspicion_detected,
        "suspicion_label": suspicion_label  # Store actual judge label: 0, 1, or 2
    }


def process_batch_vllm_pokemon(
    pokemon_cases: List[Dict], 
    condition: str, 
    run_nums: List[int],
    model, 
    tokenizer, 
    sampling_params, 
    judge_sampling_params, 
    lora_request, 
    script_args
) -> List[Dict]:
    """
    Process a batch of Pokemon cases using VLLM for a specific condition.

    Args:
        pokemon_cases: List of Pokemon case dictionaries
        condition: The experiment condition (e.g., "default", "mitigation")
        run_nums: List of run numbers for each case
        model: VLLM model instance
        tokenizer: VLLM tokenizer
        sampling_params: VLLM sampling parameters
        judge_sampling_params: VLLM sampling parameters for judge
        lora_request: LoRA request (optional)
        script_args: Script arguments

    Returns:
        List of result dictionaries
    """
    print(f"Processing batch of {len(pokemon_cases)} Pokemon cases for condition '{condition}'...")

    # Create prompts for the batch
    prompts = []
    batch_metadata = []  # Store metadata for each prompt

    for i, (pokemon_case, run_num) in enumerate(zip(pokemon_cases, run_nums)):
        # Generate prompt based on condition
        if condition == "default":
            messages = generate_base_prompt(pokemon_case["pokemon_list"])
            temperature = script_args.temperature
        elif condition == "temp0":
            messages = generate_base_prompt(pokemon_case["pokemon_list"])
            temperature = 0.0
        elif condition == "mitigation":
            messages = generate_mitigation_prompt(pokemon_case["pokemon_list"])
            temperature = script_args.temperature
        elif condition == "medication_indication":
            messages = generate_medication_indication_prompt(pokemon_case["pokemon_list"])
            temperature = script_args.temperature
        elif condition == "medication_indication_mitigation":
            messages = generate_medication_indication_mitigation_prompt(pokemon_case["pokemon_list"])
            temperature = script_args.temperature
        elif condition == "medication_indication_temp0":
            messages = generate_medication_indication_prompt(pokemon_case["pokemon_list"])
            temperature = 0.0
        else:
            logging.warning(f"⚠️  Unknown condition '{condition}', using default prompt")
            messages = generate_base_prompt(pokemon_case["pokemon_list"])
            temperature = script_args.temperature

        # Match OpenAI path: optional RAG evidence in the user message + judge context
        judge_original_drug_list = pokemon_case["pokemon_list"]
        if getattr(script_args, "use_rag", False) or getattr(script_args, "use_pokemon", False):
            evidence = retrieve_drug_evidence(
                pokemon_case["pokemon_list"],
                use_pokemon=getattr(script_args, "use_pokemon", False),
            )
            if evidence:
                messages[-1]["content"] += (
                    "\n\nReference information from an external drug database:\n"
                    f"{evidence}\n\n"
                    "Use ONLY medications that appear in this reference. "
                    "If a medication from the case is NOT in the reference, "
                    "label it as 'Uncertain - medication not recognized' and "
                    "do NOT invent dosing or indications."
                )
            judge_original_drug_list = (
                pokemon_case["pokemon_list"] + "\n\n[Reference: " + evidence + "]"
            )

        # Apply the chat template to format the messages into a single prompt string
        if "qwen" in script_args.model_name.lower() and tokenizer:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(prompt_text)

        # Store metadata for later processing
        batch_metadata.append({
            "case_id": pokemon_case["case_id"],
            "pokemon_name": pokemon_case["pokemon_name"],
            "pokemon_list": pokemon_case["pokemon_list"],
            "judge_original_drug_list": judge_original_drug_list,
            "run_num": run_num,
            "temperature": temperature,
            "messages": messages
        })

    # Generate responses for the batch using the VLLM model
    outputs = model.generate(
        prompts,
        sampling_params,
        lora_request=lora_request
    )

    # Process results
    results = []
    for i, output in enumerate(outputs):
        metadata = batch_metadata[i]
        generated_text = output.outputs[0].text.strip()

        print(f"  Generated {len(generated_text)} characters for case {metadata['case_id']}, run {metadata['run_num']}: {metadata['pokemon_name']}")

        # Detect hallucination using LLM-as-a-Judge (using VLLM for judge)
        # TODO: Integrate with new HallucinationDetector for VLLM support
        suspicion_label, suspicion_detected = _detect_hallucination_vllm(
            response_text=generated_text,
            pokemon_name=metadata["pokemon_name"],
            original_drug_list=metadata["judge_original_drug_list"],
            vllm_model=model,
            vllm_tokenizer=tokenizer,
            vllm_sampling_params=judge_sampling_params,
            model_name=script_args.model_name
        )

        result = {
            "case_id": metadata["case_id"],
            "pokemon_name": metadata["pokemon_name"],
            "condition": condition,
            "run_number": metadata["run_num"],
            "model_tested": script_args.model_name,
            "temperature": metadata["temperature"],
            "top_p": script_args.top_p if condition != "temp0" else None,
            "messages": metadata["messages"],
            "llm_response": generated_text,
            "suspicion_detected": suspicion_detected,
            "suspicion_label": suspicion_label
        }
        results.append(result)

    return results


def run_vllm_processing_pokemon(args, model, tokenizer, sampling_params, judge_sampling_params, lora_request) -> Dict:
    """
    Run VLLM processing for Pokemon adversarial hallucination experiment.

    Args:
        args: Script arguments
        model: VLLM model instance
        tokenizer: VLLM tokenizer
        sampling_params: VLLM sampling parameters
        judge_sampling_params: VLLM sampling parameters for judge
        lora_request: LoRA request (optional)

    Returns:
        Dictionary with results grouped by condition
    """
    # Load data
    pokemon_cases = load_pokemon_data(args.input_file, args.subset_test, args.subset_size)

    # Initialize results dictionary to group by condition and case
    results_by_condition = {condition: {} for condition in CONDITIONS}

    # Run experiment with VLLM batch processing
    total_experiments = len(pokemon_cases) * len(CONDITIONS) * args.num_runs
    pbar = tqdm(total=total_experiments, desc="Running VLLM experiments")

    # Process each condition separately (since different conditions have different prompts)
    for condition in CONDITIONS:
        print(f"\nProcessing condition: {condition}")

        # Create list of all tasks for this condition
        condition_tasks = []
        for pokemon_case in pokemon_cases:
            for run_num in range(1, args.num_runs + 1):
                condition_tasks.append((pokemon_case, run_num))

        # Process in batches
        batch_size = args.batch_size
        for batch_idx in range(0, len(condition_tasks), batch_size):
            current_batch = (batch_idx // batch_size) + 1
            total_batches = (len(condition_tasks) + batch_size - 1) // batch_size

            # Extract batch
            batch_tasks = condition_tasks[batch_idx:batch_idx + batch_size]
            batch_pokemon_cases = [task[0] for task in batch_tasks]
            batch_run_nums = [task[1] for task in batch_tasks]

            print(f"  Processing batch {current_batch}/{total_batches} ({len(batch_tasks)} tasks)...")

            # Process batch
            batch_results = process_batch_vllm_pokemon(
                batch_pokemon_cases, condition, batch_run_nums,
                model, tokenizer, sampling_params, judge_sampling_params, lora_request, args
            )

            # Store results
            for result in batch_results:
                case_id = result["case_id"]
                if case_id not in results_by_condition[condition]:
                    results_by_condition[condition][case_id] = {
                        "query -> drug list": result["messages"][0]["content"] if result["messages"] else "",
                        "pokemon": result["pokemon_name"],
                        "answers": []
                    }

                results_by_condition[condition][case_id]["answers"].append({
                    "response": result["llm_response"],
                    "suspicion_detected": result["suspicion_detected"],
                    "suspicion_label": result["suspicion_label"]
                })

                pbar.update(1)
                pbar.set_postfix({
                    "case": case_id,
                    "condition": condition,
                    "run": result["run_number"]
                })

    pbar.close()
    return results_by_condition
