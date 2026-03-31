import ijson
from pathlib import Path
from typing import Generator, Dict, Any


def load_alpaca_data(path: str) -> Generator[Dict[str, Any], None, None]:
    """
    Load Alpaca format JSON data from file using streaming
    
    Uses ijson for memory-efficient processing of large JSON files.
    
    Alpaca format:
    - instruction: User instruction (required)
    - input: User input (optional)
    - output: Model response (required)
    - system: System prompt (optional)
    - history: Conversation history (optional)
    
    Args:
        path: Path to the JSON file
        
    Yields:
        Dictionary with Alpaca format keys (instruction, input, output, system, history)
        
    Raises:
        ValueError: If data format is invalid or missing required keys
        FileNotFoundError: If the file does not exist
        ijson.JSONError: If the file contains invalid JSON
    """
    with open(Path(path).resolve(), 'r', encoding='utf-8') as file:
        parser = ijson.items(file, 'item')
        for i, item in enumerate(parser):
            if not isinstance(item, dict):
                raise ValueError(f"Expected item at index {i} to be a dictionary, got {type(item).__name__}")
            
            if 'instruction' not in item:
                raise ValueError(f"Missing required 'instruction' key in item at index {i}")
            
            if 'output' not in item:
                raise ValueError(f"Missing required 'output' key in item at index {i}")
            
            # Return Alpaca format directly
            yield item

