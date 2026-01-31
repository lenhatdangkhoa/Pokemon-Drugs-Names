"""
Data loading utilities for Pokemon experiment.
"""

import logging
import os
import pandas as pd
from typing import List, Dict
from pathlib import Path


def load_pokemon_data(input_file: str, subset_test: bool = False, subset_size: int = 10) -> List[Dict]:
    """Load Pokemon data from CSV.
    
    Resolves input_file path relative to the pokemon directory if it's a relative path.
    """
    # Resolve path relative to pokemon directory if it's a relative path
    if not os.path.isabs(input_file):
        # Get the pokemon directory (parent of src/)
        script_dir = Path(__file__).parent.parent
        resolved_path = script_dir / input_file
        # Normalize the path (resolve .. and .)
        input_file = str(resolved_path.resolve())
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Extract relevant columns
    data = []
    for idx, row in df.iterrows():
        data.append({
            "case_id": idx + 1,
            "pokemon_list": row.get("pokemon list", ""),
            "pokemon_name": row.get("Pokemon", "").strip()
        })
    
    if subset_test:
        data = data[:subset_size]
        logging.info(f"Using subset of {len(data)} cases for testing")
    
    logging.info(f"Loaded {len(data)} Pokemon cases")
    return data
