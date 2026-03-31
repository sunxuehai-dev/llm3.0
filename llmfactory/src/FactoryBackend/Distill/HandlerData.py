import json
from pathlib import Path
from typing import Generator, Dict, Any
from .DataSupported import load_alpaca_data
from .Response import AbstractResponse


class HandlerData:
    def __init__(self, type: str, dataset_path_or_name: str) -> None:
        """
        Initialize data handler
        
        Args:
            type: Data type (e.g., 'json', 'alpaca')
            dataset_path_or_name: Path to the dataset file
        """
        self._type = type
        self._dataset_path_or_name = dataset_path_or_name

        if type == 'alpaca':
            self._dataset: Generator[Dict[str, Any], None, None] = load_alpaca_data(dataset_path_or_name)

    def gen(self, target_path: Path, ai_response: AbstractResponse) -> None:
        """
        Generate responses for dataset items and save to target path
        
        Args:
            target_path: Path to save the generated data
            ai_response: AbstractResponse instance for generating AI responses
            
        Raises:
            ValueError: If type is not supported
            IOError: If file writing fails
        """
        if self._type == 'alpaca':
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the updated dataset to target path using streaming
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write('[\n')
                first_item = True
                for item in self._dataset:
                    # Combine instruction and input for AI response
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    combined_input = f"{instruction}\n\n{input_text}" if input_text else instruction
                    
                    # Generate AI response
                    output = ai_response.reply(combined_input)
                    
                    # Update output field
                    item['output'] = output
                    
                    # Add comma separator for all items except the first one
                    if not first_item:
                        f.write(',\n')
                    first_item = False
                    
                    # Write the item as JSON
                    json.dump(item, f, ensure_ascii=False, indent=2)
                
                f.write('\n]')
        else:
            raise ValueError(f"Unsupported type: {self._type}")