"""
项目统一时间约定：
- 业务与 JWT、数据库存储与比较：一律按 UTC（timezone-aware / naive 按 UTC 解释）。
- API JSON 中的时间字符串：由 ``API_RESPONSE_DATETIME_TZ`` 决定展示时区（默认 Asia/Shanghai），
  避免前端未做时区转换时出现「差 8 小时」的观感；需严格 UTC 时在 .env 设 ``API_RESPONSE_DATETIME_TZ=UTC``。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    """当前时刻（UTC，timezone-aware）。替代已弃用的 datetime.utcnow()。"""
    return datetime.now(timezone.utc)


def ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """将 datetime 规范为 UTC：naive 视为 UTC；aware 则转换到 UTC。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_utc_iso_z(dt: Optional[datetime]) -> Optional[str]:
    """强制 UTC + Z（与 API_RESPONSE_DATETIME_TZ 无关），仅用于需固定 UTC 字符串的场景。"""
    u = ensure_utc(dt)
    if u is None:
        return None
    return u.isoformat().replace("+00:00", "Z")


def serialize_datetime_for_api_response(dt: Optional[datetime]) -> Optional[str]:
    """
    按配置将 UTC 时刻格式化为 API JSON 字符串。
    - API_RESPONSE_DATETIME_TZ=UTC 或未配置有效时区：``...Z``
    - 例如 Asia/Shanghai：``...+08:00``（墙钟时间与北京时间一致）
    """
    if dt is None:
        return None
    u = ensure_utc(dt)
    assert u is not None
    try:
        from app.config import get_settings

        tz_name = (get_settings().API_RESPONSE_DATETIME_TZ or "").strip()
    except Exception:
        tz_name = "UTC"
    if not tz_name or tz_name.upper() == "UTC":
        return u.isoformat().replace("+00:00", "Z")
    try:
        local = u.astimezone(ZoneInfo(tz_name))
        return local.isoformat()
    except Exception:
        return u.isoformat().replace("+00:00", "Z")
