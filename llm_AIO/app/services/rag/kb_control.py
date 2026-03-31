"""RAG 向量知识库：运行时禁用开关（存用户库 SQLite）。"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from sqlalchemy.orm import Session

from app.models.user import RagVectorKbControl

ROW_ID = 1


def get_or_create_control_row(db: Session) -> RagVectorKbControl:
    row = db.get(RagVectorKbControl, ROW_ID)
    if row is None:
        row = RagVectorKbControl(id=ROW_ID, vector_kb_disabled=False)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def is_vector_kb_disabled(db: Session) -> bool:
    return get_or_create_control_row(db).vector_kb_disabled


def get_vector_kb_control_state(db: Session) -> dict[str, Any]:
    row = get_or_create_control_row(db)
    return {
        "vector_kb_disabled": row.vector_kb_disabled,
        "note": row.note,
        "updated_by_id": row.updated_by_id,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def set_vector_kb_disabled(
    db: Session,
    disabled: bool,
    note: Optional[str],
    updated_by_id: Optional[int],
) -> Tuple[dict[str, Any], str]:
    """
    更新禁用状态。返回 (state_dict, audit_message)。
    """
    row = get_or_create_control_row(db)
    row.vector_kb_disabled = disabled
    row.note = note
    row.updated_by_id = updated_by_id
    db.add(row)
    db.commit()
    db.refresh(row)
    action = "已禁用向量知识库（暂停入库与对话）" if disabled else "已启用向量知识库"
    return get_vector_kb_control_state(db), action
