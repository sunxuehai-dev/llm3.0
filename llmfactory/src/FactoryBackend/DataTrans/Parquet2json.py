import json
import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq
import fast_converter


def Parquet2json(
    target_path: Path,
    output_path: Path,
    map: dict,
    type: str,
    format: str,
    chunk_size: int = 50000
):
    """
    此函数用来转换数据格式，支持动态列名映射和多列合并。

    支持大文件流式处理，避免内存溢出。

    Args:
        target_path (Path): *.parquet 文件路径
        output_path (Path): 转换的json 文件路径
        map (dict): 列名映射字典。
            格式: { "目标字段名": "源列名" } 或 { "目标字段名": ["源列名1", "源列名2"] }
            示例: {"input": "problem"} 或 {"input": ["problem", "solution"]}
            如果是列表，会将多列数据用换行符 ('\n') 拼接。
        type (str): 任务类型 (如 'sft', 'pretrain' 等)。
        format (str): 数据格式 (如 'alpaca', 'sharegpt' 等)。

    accelerated version with PyArrow Direct Access optimization.

    比之前的json序列化版本快了6倍左右，之前做保留，以防万一
    """
    if not target_path.exists():
        raise FileNotFoundError(f"Source: {target_path}")

    if not isinstance(map, dict):
        raise ValueError(f"Map argument must be a dictionary, got {type(map)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_columns = set()
    for spec in map.values():
        if isinstance(spec, str):
            source_columns.add(spec)
        elif isinstance(spec, list):
            source_columns.update(spec)

    columns_to_read = list(source_columns)

    with open(output_path, 'w', encoding='utf-8') as f:
        pass

    parquet_file = pq.ParquetFile(target_path)
    first_chunk = True
    total_count = 0

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns_to_read):
        num_rows = batch.num_rows
        if num_rows == 0:
            continue

        columns_data = {}
        for col in batch.schema.names:
            # FIXME: 显式允许 copy，解决 "zero_copy_only was True" 错误
            # 对于字符串列，这会生成 Python Object 数组，比 Pandas 依然快得多
            columns_data[col] = batch.column(col).to_numpy(zero_copy_only=False)

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

    # C++ handles all commas and the opening bracket.
    # Python just needs to close the JSON array at the very end.
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write('\n]')

    return total_count



# NOTE: Test
# def Parquet2json(
#         target_path: Path,
#         output_path: Path,
#         map: dict,
#         type: str,
#         format: str):
#     """
#     此函数用来转换数据格式，支持动态列名映射和多列合并。

#     支持大文件流式处理，避免内存溢出。

#     Args:
#         target_path (Path): *.parquet 文件路径
#         output_path (Path): 转换的json 文件路径
#         map (dict): 列名映射字典。
#             格式: { "目标字段名": "源列名" } 或 { "目标字段名": ["源列名1", "源列名2"] }
#             示例: {"input": "problem"} 或 {"input": ["problem", "solution"]}
#             如果是列表，会将多列数据用换行符 ('\n') 拼接。
#         type (str): 任务类型 (如 'sft', 'pretrain' 等)。
#         format (str): 数据格式 (如 'alpaca', 'sharegpt' 等)。
#     """
#     # Define chunk size to balance IO and memory usage
#     CHUNK_SIZE = 1000

#     # Validate input path
#     if not target_path.exists():
#         raise FileNotFoundError(f"Target file not found: {target_path}")

#     # Validate map
#     if not isinstance(map, dict):
#         raise ValueError(f"Map argument must be a dictionary, got {type(map)}")

#     # Create output directory if it doesn't exist
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(output_path, 'w', encoding='utf-8') as f:
#         # Start the JSON array
#         f.write('[\n')

#         first_chunk = True

#         # Read the parquet file in chunks
#         for chunk in pd.read_parquet(target_path, chunksize=CHUNK_SIZE):
#             # Replace NaN with None to ensure valid JSON (NaN is not standard JSON)
#             chunk = chunk.where(pd.notnull(chunk), None)

#             # Convert the DataFrame chunk to a list of dictionaries (records)
#             records = chunk.to_dict(orient='records')

#             if not records:
#                 continue

#             # Transform records based on the map configuration
#             transformed_records = []
#             for record in records:
#                 new_record = {}
#                 for target_key, source_spec in map.items():
#                     if isinstance(source_spec, str):
#                         # Simple 1-to-1 mapping
#                         new_record[target_key] = record.get(source_spec)
#                     elif isinstance(source_spec, list):
#                         # Many-to-1 mapping (merge columns)
#                         # Filter out None values and join with newline
#                         parts = [str(record.get(col)) for col in source_spec if record.get(col) is not None]
#                         new_record[target_key] = "\n".join(parts) if parts else None
#                     else:
#                         # Invalid map format for this key
#                         new_record[target_key] = None
#                 transformed_records.append(new_record)

#             # Convert the list of dicts to a JSON string
#             # ensure_ascii=False allows proper saving of Chinese/Unicode characters
#             # indent='\t' adds tabs for readability
#             json_str = json.dumps(transformed_records, ensure_ascii=False, indent='\t')

#             # Remove the surrounding brackets [ and ] from the chunk string
#             # and strip leading/trailing newlines to ensure clean chunk joining
#             content_to_write = json_str[1:-1].strip('\n')

#             # Add a comma and a newline if this is not the first chunk
#             if not first_chunk:
#                 f.write(',\n')

#             # Write the chunk content
#             f.write(content_to_write)
#             first_chunk = False

#         # End the JSON array
#         f.write('\n]')
