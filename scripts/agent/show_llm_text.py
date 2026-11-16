"""LLM에 나가는 글 확인 (11/13) — 개인정보가 가려지는지 눈으로 보는 스크립트. LLM은 부르지 않는다 (키 · 네트워크 필요 없음).

    python scripts/agent/show_llm_text.py rx8_01                 # 가상 처방전 하나: LLM에 보낼 글을 그대로 보여 준다
    python scripts/agent/show_llm_text.py rx8_01 --inject        # 표 안에 가짜 주민번호 · 전화 · 이름 줄을 섞어 넣고 가려지는지 본다
    python scripts/agent/show_llm_text.py data/ocr_photo/rx8p_03.json   # CV가 만든 A.1 파일도 경로로 줄 수 있다
    python scripts/agent/show_llm_text.py --all                  # 샘플 전부: 가린 것이 있는 장만 자세히

보여 주는 것
  1. OCR 전체 줄 수와 그중 LLM에 보내는 줄 (약품 표 범위만)
  2. 가리기(redact.py)가 한 일: 뺀 줄 · 가린 곳의 수
  3. LLM에 나가는 글 그대로 ('[가림]'이 보이면 거기서 가린 것)
  4. 마지막 검사: 나가는 글에 주민번호 · 전화번호 모양이 하나라도 남았으면 '남음'으로 알리고 1로 끝난다

--inject의 번호는 지어낸 것이다 (900101-1234567, 010-0000-0000, 김가상(000000101)).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import RESOURCES_DIR  # noqa: E402
from agent.errors import StructureError  # noqa: E402
from agent.redact import MASK  # noqa: E402
from agent.structure import find_table, format_llm_text, prepare_llm_input  # noqa: E402

SAMPLES = RESOURCES_DIR / "samples"
FOLDERS = ("rx8_a1", "cv_a1", "a1_ocr")
Out = Callable[[str], None]

# 마지막 검사용 (redact.py와 따로 적은 느슨한 모양): 이런 글자가 나가는 글에 남으면 안 된다
LEFTOVER = {
    "주민등록번호": re.compile(r"\d{6}\s*\D?\s*[1-8][\d*]{6}"),
    "전화번호": re.compile(r"0\d{1,2}\D{1,3}\d{3,4}\D{1,3}\d{4}"),
}
FAKE_ROWS = ("주민등록번호 900101-1234567", "보호자 연락처 010-0000-0000", "김가상(000000101)")


def find(name: str) -> Path:
    path = Path(name)
    if path.suffix == ".json" and path.exists():
        return path
    for folder in FOLDERS:
        candidate = SAMPLES / folder / f"{name}.json"
        if candidate.exists():
            return candidate
    raise SystemExit(f"찾을 수 없습니다: {name} (샘플 이름 rx8_01 · rx_001 · s01_combo_a 또는 A.1 파일 경로)")


def inject(ocr: dict[str, Any]) -> dict[str, Any]:
    """약품 표 안(첫 약품 줄 바로 아래)에 가짜 개인정보 줄을 섞는다. 표 범위를 잘못 잡은 상황을 흉내 낸다."""
    ocr = copy.deepcopy(ocr)
    lines = ocr["lines"]
    try:
        top, _ = find_table(lines)
    except StructureError:
        top = min(l["bbox"][1] for l in lines)
    height = 40
    x1 = min(l["bbox"][0] for l in lines)
    start = max(l.get("line_id", i) for i, l in enumerate(lines)) + 1
    for k, text in enumerate(FAKE_ROWS):
        y = top + 4 + k * 3          # 표 머리글 바로 아래, 다른 줄과 같은 높이로 붙인다
        lines.append({"line_id": start + k, "text": text, "bbox": [x1 + 1500, y, x1 + 2000, y + height], "conf": 0.9})
    return ocr


def leftovers(text: str) -> list[str]:
    return [kind for kind, pattern in LEFTOVER.items() if pattern.search(text)]


def show(name: str, ocr: dict[str, Any], out: Out, *, brief: bool = False) -> bool:
    """한 장을 보여 준다. 나가는 글에 개인정보 모양이 남지 않았으면 True."""
    issue_no, rows, changed = prepare_llm_input(ocr)
    text = format_llm_text(issue_no, rows)
    left = leftovers(text)
    masks = text.count(MASK)
    out(f"[{name}] OCR {len(ocr['lines'])}칸 → LLM에 보낼 줄 {len(rows)}개 | 가리기: {changed}곳"
        f"{f' (글 안의 [가림] {masks}곳, 나머지는 줄째로 뺌)' if changed else ''} | 검사: {'남음 ' + ', '.join(left) if left else '개인정보 모양 없음'}")
    if brief and not changed and not left:
        return True
    out("    ---- LLM에 나가는 글 ----")
    for line in text.splitlines():
        out(f"    {line}")
    out("    ------------------------")
    return not left


def main(argv: list[str] | None = None, *, out: Out = print) -> int:
    parser = argparse.ArgumentParser(description="LLM에 나가는 글 확인 (개인정보 가리기)")
    parser.add_argument("sample", nargs="?", default="rx8_01", help="샘플 이름 또는 A.1 JSON 경로 (기본 %(default)s)")
    parser.add_argument("--inject", action="store_true", help="표 안에 가짜 주민번호 · 전화 · 이름 줄을 섞는다")
    parser.add_argument("--all", action="store_true", help="샘플 전부 (가린 것이 있는 장만 글을 보여 준다)")
    args = parser.parse_args(argv)

    if args.all:
        paths = [p for f in FOLDERS for p in sorted((SAMPLES / f).glob("*.json"))
                 if "lines" in json.loads(p.read_text(encoding="utf-8"))]
    else:
        paths = [find(args.sample)]
    clean = True
    for path in paths:
        ocr = json.loads(path.read_text(encoding="utf-8"))
        if args.inject:
            ocr = inject(ocr)
            out(f"(가짜 개인정보 줄을 표 안에 넣었습니다: {' / '.join(FAKE_ROWS)})")
        try:
            clean &= show(path.stem, ocr, out, brief=args.all)
        except StructureError as e:
            out(f"[{path.stem}] 읽을 수 없음: {e}")
    out("결과: 나가는 글에 개인정보 모양이 없습니다." if clean else "결과: 나가는 글에 개인정보 모양이 남았습니다. 알려 주세요.")
    return 0 if clean else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
