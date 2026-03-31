import pyarrow.dataset as ds
from pathlib import Path
import fast_converter


def Arrow2json(
        target_path: Path,
        output_path: Path,
        map: dict,
        type: str,
        format: str,
        chunk_size: int = 10000
):
    """
    此函数用来转换数据格式，支持动态列名映射和多列合并。
    使用 C++ 加速模块进行转换，速度极快。
    处理 Arrow (.arrow) 文件。

    Args:
        target_path (Path): *.arrow 文件路径
        output_path (Path): 转换的json 文件路径
        map (dict): 列名映射字典。
            格式: { "目标字段名": "源列名" } 或 { "目标字段名": ["源列名1", "源列名2"] }
            示例: {"input": "problem"} 或 {"input": ["problem", "solution"]}
            如果是列表，会将多列数据用换行符 ('\n') 拼接。
        type (str): 任务类型 (如 'sft', 'pretrain' 等)。
        format (str): 数据格式 (如 'alpaca', 'sharegpt' 等)。
        chunk_size (int): 每次处理的批次大小。
    """
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    if not isinstance(map, dict):
        raise ValueError(f"Map argument must be a dictionary, got {type(map)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare list of columns to read from the file (optimization)
    source_columns = set()
    for spec in map.values():
        if isinstance(spec, str):
            source_columns.add(spec)
        elif isinstance(spec, list):
            source_columns.update(spec)

    columns_to_read = list(source_columns)

    with open(output_path, 'w', encoding='utf-8') as f:
        pass

    dataset = ds.dataset(target_path, format='ipc')
    first_chunk = True
    total_count = 0

    for batch in dataset.to_batches(batch_size=chunk_size, columns=columns_to_read):
        num_rows = batch.num_rows
        if num_rows == 0:
            continue

        columns_data = {}
        for col in batch.schema.names:
            columns_data[col] = batch.column(col).to_numpy(zero_copy_only=False)

        # Call the C++ accelerator to process and write the chunk
        fast_converter.process_chunk_to_json(
            columns_data,
            map,
            str(output_path),
            first_chunk,
            num_rows,
        )

        total_count += num_rows
        first_chunk = False

        print(f"  > Processed {total_count} rows...", end='\r')

    # C++ handles commas and the opening bracket.
    # Python just needs to close the JSON array at the very end.
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write('\n]')

    return total_count
