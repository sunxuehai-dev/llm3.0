from .Parquet2json import Parquet2json
from .Arrow2json import Arrow2json

from pathlib import Path
import json

def gen_dataset_parquet2json(
    dataset_name: str,
    dataset_dir: Path,
    output_dir: Path,
    col_map: dict,
    task_type: str = 'sft',
    data_format:str = 'alpaca'
    ) -> list[str]:
    """
    对Parquet2json函数的进一步封装
    output_dir下会自动生成dataset_info.json
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    block_index = 0
    gen_files_list = []
    dataset_info = {}

    parquet_files = sorted(dataset_dir.rglob('*.parquet'))

    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name} -> {dataset_name}_{block_index}.json")

        output_file = output_dir / f'{dataset_name}_{block_index}.json'

        Parquet2json(
            target_path=parquet_file,
            output_path=output_file,
            map=col_map,
            type=task_type,
            format=data_format
        )

        dataset_key = f'{dataset_name}_{block_index}'
        dataset_info[dataset_key] = {
            "file_name": f'{dataset_name}_{block_index}.json'
        }
        gen_files_list.append(dataset_key)
        
        block_index += 1
    
    dataset_info_path = output_dir / 'dataset_info.json'
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    return gen_files_list