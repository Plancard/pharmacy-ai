"""업로드 이미지 저장소. DB에는 storage_key만 넣는다 (ERD 설계 규칙 7).

1차: VAR_DIR 아래 파일로 저장. 2차에 객체 저장소(S3·MinIO)로 바꿀 때 이 파일의 함수만 바꾼다.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import current_app

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def _root() -> Path:
    return Path(current_app.config["VAR_DIR"])


def save(prescription_id: str, data: bytes, ext: str) -> str:
    """이미지 바이트를 저장하고 storage_key를 돌려준다. 예: uploads/<처방 ID>/<uuid>.jpg"""
    ext = ext.lower() if ext.startswith(".") else "." + ext.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"허용하지 않는 이미지 형식입니다: {ext}")
    key = f"uploads/{uuid.UUID(str(prescription_id))}/{uuid.uuid4().hex}{ext}"
    path = path_of(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return key


def path_of(key: str) -> Path:
    """storage_key → 실제 파일 경로. VAR_DIR 밖을 가리키는 키는 거부한다."""
    root = _root().resolve()
    path = (root / key).resolve()
    if root not in path.parents:
        raise ValueError("잘못된 저장소 키입니다.")
    return path


def read(key: str) -> bytes:
    return path_of(key).read_bytes()


def overlay_path(path: Path) -> Path:
    """트레이 사진 파일 → 검수 결과 오버레이 그림 파일 (같은 폴더, 이름 뒤에 .overlay.jpg, cv/pipeline.py가 만든다).

    처방전의 펴진 이미지(.rect.jpg)와 같은 파생 파일 방식이다. 검수 전이거나 뒤집힘 판정·저장 실패면 파일이 없다.
    """
    return path.with_name(path.name + ".overlay.jpg")
