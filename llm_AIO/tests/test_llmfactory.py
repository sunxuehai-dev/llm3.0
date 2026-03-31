"""
llmfactory 集成接口测试
测试 LoRA/全量微调、合并适配器、推理 API 启动等接口

注意：
- 异步接口（/train/lora、/train/full、/merge）仅验证任务提交成功，不等待实际完成
- 同步接口（/train/lora/sync 等）若路径无效会快速失败，用于验证 API 层逻辑
- 完整训练/合并需配置真实模型与数据集路径
"""
import os
import json
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://localhost:8000"
LLMFACTORY_BASE = f"{BASE_URL}/api/playground/llmfactory"


def print_result(test_name: str, success: bool, message: str = "", data=None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data is not None:
        try:
            print(f"      响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
        except Exception:
            print(f"      响应: {data}")
    print()


def test_health():
    """服务健康检查"""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        ok = r.status_code == 200
        print_result("健康检查", ok, f"状态码: {r.status_code}", r.json() if ok else r.text)
        return ok
    except Exception as e:
        print_result("健康检查", False, f"连接失败: {e}")
        return False


def test_train_lora_empty_body():
    """LoRA 训练 - 空请求体（依赖 .env 默认值或返回 400）"""
    r = requests.post(f"{LLMFACTORY_BASE}/train/lora", json={}, timeout=10)
    # 有 .env 默认值时 200，无默认值时 400
    ok = r.status_code in (200, 400)
    detail = r.json().get("detail", "") if r.status_code != 200 else r.json().get("message", "")
    print_result(
        "LoRA 训练 - 空请求体",
        ok,
        f"状态码: {r.status_code}, 预期: 200(有默认值) 或 400(缺少参数)",
        r.json(),
    )
    return ok


def test_train_lora_with_params():
    """LoRA 训练 - 显式传参（仅验证接口接受请求）"""
    payload = {
        "model_path": "/tmp/test_model",
        "dataset": "identity,alpaca_en_demo",
        "dataset_dir": "/tmp/test_data",
        "output_dir": "/tmp/test_output",
        "template": "qwen3_nothink",
        "lora_rank": 8,
        "num_train_epochs": 1,
    }
    r = requests.post(f"{LLMFACTORY_BASE}/train/lora", json=payload, timeout=10)
    ok = r.status_code == 200
    body = r.json()
    print_result(
        "LoRA 训练 - 显式传参",
        ok,
        f"状态码: {r.status_code}, success={body.get('success')}",
        body,
    )
    return ok


def test_train_full_with_params():
    """全量微调 - 显式传参"""
    payload = {
        "model_path": "/tmp/test_model",
        "dataset": "identity",
        "dataset_dir": "/tmp/test_data",
        "output_dir": "/tmp/test_output",
        "num_train_epochs": 1,
    }
    r = requests.post(f"{LLMFACTORY_BASE}/train/full", json=payload, timeout=10)
    ok = r.status_code == 200
    body = r.json()
    print_result(
        "全量微调 - 显式传参",
        ok,
        f"状态码: {r.status_code}, success={body.get('success')}",
        body,
    )
    return ok


def test_merge_missing_params():
    """合并 - 缺少必要参数（应返回 400）"""
    r = requests.post(f"{LLMFACTORY_BASE}/merge", json={}, timeout=10)
    ok = r.status_code == 400
    detail = r.json().get("detail", "")
    print_result(
        "合并 - 缺少参数",
        ok,
        f"状态码: {r.status_code}, 预期 400, detail={detail[:80] if detail else '-'}",
        r.json(),
    )
    return ok


def test_merge_with_params():
    """合并 - 显式传参（仅验证接口接受）"""
    payload = {
        "model_path": "/tmp/test_model",
        "adapter_path": "/tmp/test_adapter",
        "export_dir": "/tmp/test_export",
    }
    r = requests.post(f"{LLMFACTORY_BASE}/merge", json=payload, timeout=10)
    ok = r.status_code == 200
    body = r.json()
    print_result(
        "合并 - 显式传参",
        ok,
        f"状态码: {r.status_code}, success={body.get('success')}",
        body,
    )
    return ok


def test_train_lora_sync_invalid_paths():
    """LoRA 同步训练 - 无效路径（验证 API 接受请求，子进程会快速失败）"""
    payload = {
        "model_path": "/nonexistent/model",
        "dataset": "identity",
        "dataset_dir": "/nonexistent/data",
        "output_dir": "/tmp/test_output",
        "num_train_epochs": 1,
    }
    r = requests.post(f"{LLMFACTORY_BASE}/train/lora/sync", json=payload, timeout=60)
    # 可能 200(return_code!=0) 或 500(异常)
    ok = r.status_code in (200, 500)
    body = r.json() if r.status_code == 200 else {"detail": r.text[:100]}
    print_result(
        "LoRA 同步训练 - 无效路径",
        ok,
        f"状态码: {r.status_code}, 预期接受请求(200或500)",
        body,
    )
    return ok


def test_merge_sync_invalid_paths():
    """合并同步 - 无效路径"""
    payload = {
        "model_path": "/nonexistent/model",
        "adapter_path": "/nonexistent/adapter",
        "export_dir": "/tmp/test_export",
    }
    r = requests.post(f"{LLMFACTORY_BASE}/merge/sync", json=payload, timeout=60)
    ok = r.status_code in (200, 500)
    body = r.json() if r.status_code == 200 else {"detail": r.text[:100]}
    print_result(
        "合并同步 - 无效路径",
        ok,
        f"状态码: {r.status_code}",
        body,
    )
    return ok


def test_start_inference_api_missing_model():
    """启动推理 API - 缺少 model_path（应返回 400）"""
    r = requests.post(f"{LLMFACTORY_BASE}/api/start", json={}, timeout=10)
    ok = r.status_code == 400
    print_result(
        "启动推理API - 缺少model_path",
        ok,
        f"状态码: {r.status_code}, 预期 400",
        r.json(),
    )
    return ok


def test_start_inference_api_with_model():
    """启动推理 API - 显式传 model_path（会启动子进程，谨慎执行）"""
    payload = {
        "model_path": "/tmp/test_model",
        "api_port": "18999",
    }
    r = requests.post(f"{LLMFACTORY_BASE}/api/start", json=payload, timeout=15)
    # 200 启动成功；400 路径不存在；500 其他错误
    ok = r.status_code in (200, 400, 500)
    body = r.json() if r.status_code == 200 else (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})
    print_result(
        "启动推理API - 显式传参",
        ok,
        f"状态码: {r.status_code}（200=已启动，400=路径无效，500=其他错误）",
        body,
    )
    return ok


def main():
    print("=" * 60)
    print("llmfactory 集成接口测试")
    print("=" * 60)
    print()

    if not test_health():
        print("❌ 服务未启动，请先启动: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return

    print("-" * 60)
    print("异步接口测试（任务提交即返回）")
    print("-" * 60)
    test_train_lora_empty_body()
    test_train_lora_with_params()
    test_train_full_with_params()
    test_merge_missing_params()
    test_merge_with_params()

    print("-" * 60)
    print("同步接口测试（会实际执行，无效路径会快速失败）")
    print("-" * 60)
    test_train_lora_sync_invalid_paths()
    test_merge_sync_invalid_paths()

    print("-" * 60)
    print("推理 API 启动测试")
    print("-" * 60)
    test_start_inference_api_missing_model()
    test_start_inference_api_with_model()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
