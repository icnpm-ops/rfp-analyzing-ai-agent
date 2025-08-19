# backend/embedding/metadata_store.py
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from backend.config import METADATA_PATH

def _ensure_dir():
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)

def _read_all() -> List[Dict[str, Any]]:
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_all(data: List[Dict[str, Any]]) -> None:
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_metadata(title: str, source: str, doc_type: str, doc_id: Optional[str] = None) -> None:
    """동기 저장. 같은 docID가 이미 있으면 중복 저장하지 않음."""
    _ensure_dir()
    data = _read_all()
    if doc_id and any(item.get("docID") == doc_id for item in data):
        # 이미 저장되어 있으면 스킵
        return
    entry = {
        "title": title,
        "source": source,
        "docType": doc_type,
        "uploadAt": datetime.now(timezone.utc).isoformat(),
    }
    if doc_id:
        entry["docID"] = doc_id
    data.append(entry)
    _write_all(data)
    print("[metadata] saved:", entry)

def load_all_metadata() -> List[Dict[str, Any]]:
    return _read_all()

def get_metadata_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    for item in _read_all():
        if item.get("docID") == doc_id:
            return item
    return None
