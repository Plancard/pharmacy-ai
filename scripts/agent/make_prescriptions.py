"""대상 8종으로 시연용 가상 처방전을 만든다. CV 팀 가상 처방전과 같은 양식(별지 제9호 서식)이다.

    python scripts/agent/make_prescriptions.py              # JSON(A.1 · 정답 · CV 정답)을 다시 만든다
    python scripts/agent/make_prescriptions.py --images     # 처방전 이미지(PNG)도 만든다 (Pillow와 한글 글꼴 필요)
    python scripts/agent/make_prescriptions.py --images --font C:\\Windows\\Fonts\\malgun.ttf

pytest가 아니다. 아래 PRESCRIPTIONS 표가 처방전 내용의 원본이고, 파일은 이 표에서 만든다. 처방을 바꿀 때는 표를 고치고 다시 돌린다.
  agent/resources/samples/rx8_a1/        A.1 OCR 결과 (CV가 글자를 틀림없이 읽었다고 친 모습, conf 0.99)
  agent/resources/samples/rx8_expected/  정답: 약품 줄 · 위험 신호 단계 · 조제 계획을 만들 수 있는지
  agent/resources/samples/rx8_img/       처방전 이미지 (CV가 OCR로 읽어 볼 원본, A.1과 같은 위치에 같은 글자)
                                         + ground_truth.json · gt.json (CV의 cv/ocr_test.py --gt 에 그대로 넣는 정답)

규칙
- 양식은 CV 팀 가상 처방전(cv/fixtures, rx_001~005)과 같다: 칸 위치, 머리글, 약품 줄 "(1)코드 이름/함량", 용법 "PO 식후 30분".
- 대상 8종(agent/resources/drug_dictionary.json)만 쓰고, 약품 코드 칸에는 품목기준코드를 인쇄한다.
  그래서 약품 매칭이 코드 단계에서 자동 확정된다 (실제 처방전의 보험코드 자리. 일반의약품이라 가상으로 정했다).
- 트레이 규격(명세서 2.7절)을 지킨다: 3일 이하, 한 칸(일자 x 시점)에 2알 이하. rx8_07만 일부러 한 칸에 3알이 들어가
  조제 계획이 E_PLAN_RANGE로 막히는 시연용이다.
- 용법은 실제 처방전처럼 "PO 식후 30분"이다. 시점이 없으므로 Agent가 하루 횟수로 시점을 배정하고 경고를 단다
  (2회 아침·저녁, 3회 아침·점심·저녁). 취침 전 약만 "PO 취침 전"이라 적는다.
- 환자 · 의료기관 · 의료인은 모두 가상이고 주민등록번호는 가려져 있다. 모든 처방전 아래에 '테스트용 가상 처방전' 문구가 들어간다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import config  # noqa: E402
from agent.drug_dictionary import load_dictionary  # noqa: E402

SAMPLES = config.RESOURCES_DIR / "samples"
A1_DIR, EXPECTED_DIR, IMG_DIR = SAMPLES / "rx8_a1", SAMPLES / "rx8_expected", SAMPLES / "rx8_img"
WIDTH, HEIGHT = 2480, 3507
CONF = 0.99

UNIT = {"캡슐": "CAP"}      # 이름에 '캡슐'이 들어가면 CAP, 아니면 TAB
USAGE_AFTER_MEAL, USAGE_BEDTIME = "PO 식후 30분", "PO 취침 전"
SHOWN_NAMES = {              # 처방전에 인쇄하는 이름 (CV 양식: 이름/함량). 없으면 사전 이름 그대로
    "타이레놀정500mg": "타이레놀정/500mg",
    "뮤코다인캡슐200mg": "뮤코다인캡슐/200mg",
    "아이월드요비신정 416.7mg": "아이월드요비신정/416.7mg",
}

# 처방전 표: 보여 주는 것, 기대 위험 단계, 조제 계획 결과, 가상 환자 · 의료기관, [(약품 이름, 1회 개수, 시점, 일수)]
MTN, ME, BED = ("morning", "noon", "evening"), ("morning", "evening"), ("bedtime",)
HOSPITAL, CLINIC = "(의)가상의료재단 테스트병원", "가상정형외과의원"
PRESCRIPTIONS: list[dict[str, Any]] = [
    {"id": "rx8_01", "purpose": "시연 조합 A: 아세트아미노펜 중복 (경고)", "risk": "warning", "plan": "ok",
     "dept": "가정의학과", "patient": "김가상", "disease": "R51", "doctor": "이테스트", "hospital": HOSPITAL, "time": "09:05:12",
     "drugs": [("타이레놀정500mg", 1, MTN, 3), ("게보린정", 1, MTN, 3)]},
    {"id": "rx8_02", "purpose": "시연 조합 B: 이부프로펜 중복 (경고)", "risk": "warning", "plan": "ok",
     "dept": "정형외과", "patient": "박샘플", "disease": "M54.5", "doctor": "한모의", "hospital": CLINIC, "time": "09:41:30",
     "drugs": [("트리스펜연질캡슐", 1, ME, 3), ("아르펜정", 1, ME, 3)]},
    {"id": "rx8_03", "purpose": "위험 신호 없음", "risk": "none", "plan": "ok",
     "dept": "이비인후과", "patient": "정데모", "disease": "J01.9", "doctor": "최예시", "hospital": HOSPITAL, "time": "10:17:48",
     "drugs": [("프론크논캡슐", 1, MTN, 3), ("뮤코다인캡슐200mg", 1, MTN, 3)]},
    {"id": "rx8_04", "purpose": "외형이 비슷한 타이레놀 · 트립라인 (CV 혼동 평가)", "risk": "none", "plan": "ok",
     "dept": "가정의학과", "patient": "윤시험", "disease": "R50.9", "doctor": "이테스트", "hospital": HOSPITAL, "time": "11:02:05",
     "drugs": [("타이레놀정500mg", 1, MTN, 3), ("트립라인정", 1, MTN, 3)]},
    {"id": "rx8_05", "purpose": "1회 2알 + 취침 전 (칸당 2알 꽉 채움)", "risk": "none", "plan": "ok",
     "dept": "이비인후과", "patient": "강가명", "disease": "J20.9", "doctor": "최예시", "hospital": HOSPITAL, "time": "11:36:22",
     "drugs": [("뮤코다인캡슐200mg", 2, ME, 3), ("트립라인정", 1, BED, 3)]},
    {"id": "rx8_06", "purpose": "한약제제 요비신정 + 2일 처방", "risk": "none", "plan": "ok",
     "dept": "비뇨의학과", "patient": "오예제", "disease": "N30.9", "doctor": "오가상", "hospital": HOSPITAL, "time": "13:24:57",
     "drugs": [("아이월드요비신정 416.7mg", 1, ME, 2), ("아르펜정", 1, MTN, 2)]},
    {"id": "rx8_07", "purpose": "트레이 초과: 한 칸에 3알 (조제 계획 E_PLAN_RANGE)", "risk": "warning", "plan": "E_PLAN_RANGE",
     "dept": "내과", "patient": "서보기", "disease": "J06.9", "doctor": "최예시", "hospital": HOSPITAL, "time": "14:08:33",
     "drugs": [("타이레놀정500mg", 1, MTN, 3), ("게보린정", 1, MTN, 3), ("뮤코다인캡슐200mg", 1, MTN, 3)]},
    {"id": "rx8_08", "purpose": "1일 처방 · 세 시점 (요비신정 · 트리스펜)", "risk": "none", "plan": "ok",
     "dept": "가정의학과", "patient": "임견본", "disease": "R52", "doctor": "이테스트", "hospital": HOSPITAL, "time": "15:51:09",
     "drugs": [("트리스펜연질캡슐", 1, MTN, 1), ("아이월드요비신정 416.7mg", 1, MTN, 1)]},
]
DOCTOR_LICENSE = {"이테스트": "00001", "최예시": "00002", "한모의": "00003", "오가상": "00004"}
ISSUE_DATE, PRINT_DATE = "2026년11월11일", "2026.11.11"

# ---------------------------------------------------------------- 양식 (CV 가상 처방전 rx_001~005에서 잰 위치)
# 글자: (x, y, 글자, 글자 크기 px, 굵게). 박스는 글자 수로 어림한다 (글꼴마다 폭이 달라도 같은 JSON이 나오게)
FORM_TEXT: list[tuple[int, int, str, int, bool]] = [
    (206, 146, "[별지 제9호 서식]", 38, False),
    (968, 280, "처   방   전", 108, True),
    (1884, 312, "약국제출용", 54, False),
    (296, 432, "건강보험", 40, False), (490, 432, "의료급여", 40, False), (685, 432, "산업재해보험", 40, False),
    (962, 432, "자동차보험", 40, False), (1212, 432, "기타", 40, False),
    (1860, 434, "요양기관기호", 34, False), (2070, 434, ":", 34, False), (2088, 434, "99990001", 36, False),
    (268, 516, "발급 연월일 및 번호", 40, False),
    (1570, 516, "명   칭", 40, False),
    (1432, 590, "의", 38, False), (1432, 633, "료", 38, False), (1432, 676, "기", 38, False), (1432, 719, "관", 38, False),
    (1558, 598, "전 화 번 호", 40, False), (1790, 602, "02-000-0000", 40, False),
    (1558, 680, "팩 스 번 호", 40, False), (1790, 684, "02-000-0001", 40, False),
    (1558, 762, "홈 페 이 지", 40, False), (1794, 764, "www.test-hosp.example", 40, False),
    (239, 635, "환", 40, False), (239, 728, "자", 40, False),
    (389, 617, "성   명", 40, False),
    (358, 738, "주민등록번호", 40, False), (857, 740, "900101-1******", 42, False),
    (272, 848, "질병", 34, False), (272, 888, "분류", 34, False), (272, 928, "기호", 34, False),
    (991, 852, "처  방", 32, False), (991, 891, "의료인의", 32, False), (991, 930, "성  명", 32, False),
    (1570, 847, "면허종류", 40, False), (1979, 847, "의사", 40, False),
    (1570, 929, "면허번호", 40, False), (1893, 929, "제", 40, False), (2098, 929, "호", 40, False),
    (257, 1009, "※환자의 요구가 있을 때에는 질병분류기호를 기재하지 아니합니다.", 40, False),
    (1307, 1104, "1회", 34, False), (1309, 1147, "투약량", 34, False),
    (1482, 1090, "1일", 30, False), (1480, 1124, "투여", 30, False), (1478, 1164, "횟수", 30, False),
    (1631, 1088, "총", 30, False), (1622, 1124, "투약", 30, False), (1621, 1164, "일수", 30, False),
    (1751, 1090, "본인", 30, False), (1749, 1124, "부담률", 30, False), (1751, 1164, "구분코드", 30, False),
    (592, 1124, "처방 의약품의 명칭", 40, False),
    (2006, 1124, "용", 40, False), (2100, 1124, "법", 40, False),
    (257, 2797, "주사제 처방명세([  ]원 내 조제,", 38, False), (800, 2797, "[  ]원 외 처방)", 38, False),
    (1626, 2793, "조제 시", 36, False), (1624, 2834, "참고 사항", 36, False),
    (2010, 2792, "본인부담", 36, False), (2010, 2834, "구분기호", 36, False),
    (272, 3072, "사용기간", 40, False), (637, 3072, "발급일부터(", 40, False), (880, 3072, "7", 40, False),
    (966, 3072, ")일간", 40, False), (1435, 3072, "사용기간내에 약국에 제출하여야 합니다.", 40, False),
    (1042, 3232, "의 약 품 조 제 명 세", 56, True),
    (339, 3330, "조제 기관의 명칭", 38, False), (358, 3388, "조제약사 성명", 38, False),
    (1600, 3360, "처방의 변경·수정·확인·대체시 그 내용 등", 32, False),
    (789, 3468, "※ OCR 테스트용 가상 처방전", 30, False), (1161, 3468, "실제 환자 · 의료기관 · 의료인 정보가 아닙니다", 30, False),
]
# 표 선: (x1, y1, x2, y2, 굵기)
FORM_LINES: list[tuple[int, int, int, int, int]] = [
    (212, 238, 2267, 238, 5), (212, 3139, 2267, 3139, 5), (214, 236, 214, 3141, 5), (2265, 236, 2265, 3141, 5),
    (212, 413, 2267, 413, 3), (212, 496, 2267, 496, 3), (212, 578, 1394, 578, 3), (1510, 578, 2267, 578, 3),
    (1510, 661, 2267, 661, 3), (306, 696, 1394, 696, 3), (1510, 744, 2267, 744, 3), (212, 826, 2267, 826, 3),
    (1510, 909, 2267, 909, 3), (212, 992, 2267, 992, 3), (212, 1074, 2267, 1074, 3), (212, 1216, 2267, 1216, 3),
    (212, 2775, 2267, 2775, 3), (212, 2858, 1512, 2858, 3), (1510, 2893, 2267, 2893, 3), (212, 3047, 2267, 3047, 3),
    (306, 578, 306, 827, 3), (400, 826, 400, 993, 3), (613, 496, 613, 827, 3), (943, 826, 943, 993, 3),
    (1156, 826, 1156, 993, 3), (1392, 496, 1392, 827, 3), (1510, 496, 1510, 993, 3), (1510, 2775, 1510, 3048, 3),
    (1770, 496, 1770, 993, 3), (1274, 1074, 1274, 2776, 3), (1439, 1074, 1439, 2776, 3), (1581, 1074, 1581, 2776, 3),
    (1723, 1074, 1723, 2776, 3), (1888, 1074, 1888, 3048, 3), (471, 3047, 471, 3141, 3), (1239, 3047, 1239, 3141, 3),
    (212, 3308, 2267, 3308, 3), (212, 3447, 2267, 3447, 3), (214, 3308, 214, 3448, 3), (2265, 3308, 2265, 3448, 3),
    (707, 3308, 707, 3448, 3), (1534, 3308, 1534, 3448, 3),
]
CHECKBOXES = [(456, 448), (650, 448), (928, 448), (1178, 448)]   # 빈 네모 (왼쪽 위). 건강보험 앞은 체크 표시
ROW_TOP, ROW_STEP, ROW_SIZE = 1256, 78, 42                         # 약품 줄 (CV 양식: 첫 줄만 앞에 (1))


def text_width(text: str, size: int) -> int:
    """글꼴 없이 어림한 글자 폭 (한글 1자 = 글자 크기, 영숫자 0.56, 공백 0.3)."""
    units = 0.0
    for ch in text:
        if ch == " ":
            units += 0.3
        elif ord(ch) < 0x2E80:
            units += 0.56
        else:
            units += 1.0
    return round(units * size)


def _box(x: int, y: int, text: str, size: int) -> list[int]:
    return [x, y, x + text_width(text, size), y + round(size * 1.2)]


def shown_name(name: str) -> str:
    return SHOWN_NAMES.get(name, name)


def read_name(name: str) -> str:
    """구조화가 읽어 내는 이름 (인쇄한 이름의 '/'가 공백이 된다). 약품 확정은 코드로 하므로 이름은 읽은 그대로다."""
    return " ".join(shown_name(name).replace("/", " ").split())


def form_texts(rx: dict[str, Any], number: int, codes: dict[str, str]) -> list[tuple[int, int, str, int, bool]]:
    """처방전 한 장에 찍는 글자 전부 (양식 + 이 처방전의 값)."""
    texts = list(FORM_TEXT)
    texts += [
        (298, 296, rx["dept"], 62, False),
        (775, 516, f"{ISSUE_DATE} {100 + number:05d}호", 46, False),
        (1782, 518, rx["hospital"], 38, False),
        (798, 612, f"{rx['patient']}({200 + number:09d})", 52, False),
        (467, 855, rx["disease"], 42, False),
        (1200, 886, rx["doctor"], 46, False),
        (1961, 933, DOCTOR_LICENSE[rx["doctor"]], 40, False),
        (206, 3166, f"출력시간 : {PRINT_DATE} {rx['time']}", 40, False),
    ]
    for row, (name, count, times, days) in enumerate(rx["drugs"]):
        y = ROW_TOP + row * ROW_STEP
        unit = "CAP" if "캡슐" in name else "TAB"
        label = f"{codes[name]} {shown_name(name)}"
        texts += [
            (230, y, f"(1){label}", ROW_SIZE, False) if row == 0 else (257, y, label, ROW_SIZE, False),
            (1303, y + 22, f"{count} {unit}", 40, False),
            (1478, y + 18, f"{len(times)}회", 40, False),
            (1619, y + 18, f"{days}일", 40, False),
            (1921, y + 18, USAGE_BEDTIME if tuple(times) == BED else USAGE_AFTER_MEAL, 40, False),
        ]
    return texts


def a1_of(rx: dict[str, Any], number: int, codes: dict[str, str]) -> dict[str, Any]:
    boxes = [(_box(x, y, text, size), text) for x, y, text, size, _ in form_texts(rx, number, codes)]
    boxes.sort(key=lambda b: ((b[0][1] + b[0][3]) / 2, b[0][0]))      # A.1: 박스 중심 y, x 순
    lines = [{"line_id": i, "text": text, "bbox": bbox, "conf": CONF} for i, (bbox, text) in enumerate(boxes)]
    return {"image_id": rx["id"], "doc_type": "prescription", "width": WIDTH, "height": HEIGHT, "lines": lines}


def expected_of(rx: dict[str, Any], codes: dict[str, str]) -> dict[str, Any]:
    return {
        "image_id": rx["id"],
        "purpose": rx["purpose"],
        "risk_level": rx["risk"],
        "plan": rx["plan"],
        "items": [
            {"item_code": codes[name], "name": read_name(name), "count_per_dose": count, "times": list(times), "days": days}
            for name, count, times, days in rx["drugs"]
        ],
    }


def cv_ground_truth(codes: dict[str, str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """CV 팀 정답 형식 두 가지: cv/ground_truth.json과 같은 자세한 것, cv/gt.json과 같은 약품명 목록(ocr_test.py --gt)."""
    detail, names = {}, {}
    for number, rx in enumerate(PRESCRIPTIONS, start=1):
        image = f"{rx['id']}.png"
        detail[image] = {
            "patient": f"{rx['patient']}({200 + number:09d})",
            "issue": f"{ISSUE_DATE} {100 + number:05d}호",
            "drugs": [
                {"code": codes[name], "name": shown_name(name), "dose": count, "unit": "CAP" if "캡슐" in name else "TAB",
                 "times": len(times), "days": days, "usage": USAGE_BEDTIME if tuple(times) == BED else USAGE_AFTER_MEAL}
                for name, count, times, days in rx["drugs"]
            ],
        }
        names[image] = [shown_name(d[0]) for d in rx["drugs"]]
    return detail, names


def draw(rx: dict[str, Any], number: int, codes: dict[str, str], path: Path, font_path: str, bold_path: str | None = None) -> None:
    from PIL import Image, ImageDraw, ImageFont   # 이미지를 만들 때만 필요하다

    image = Image.new("L", (WIDTH, HEIGHT), 255)
    pen = ImageDraw.Draw(image)
    for x1, y1, x2, y2, width in FORM_LINES:
        pen.line([x1, y1, x2, y2], fill=0, width=width)
    for x, y in CHECKBOXES:
        pen.rectangle([x, y, x + 26, y + 26], outline=0, width=2)
    pen.line([258, 462, 268, 478, 286, 440], fill=0, width=3)                 # 건강보험 체크
    fonts: dict[tuple[int, bool], Any] = {}
    for x, y, text, size, bold in form_texts(rx, number, codes):
        key = (size, bold)
        if key not in fonts:
            fonts[key] = ImageFont.truetype(bold_path if bold and bold_path else font_path, size)
        gray = 110 if y > 3460 else 0                                         # 맨 아래 안내 문구는 회색
        pen.text((x, y), text, fill=gray, font=fonts[key])
    image.save(path, optimize=True)


DEFAULT_FONTS = [   # (보통, 굵게)
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", None),
]


def main(argv: list[str] | None = None, *, out=print) -> int:
    parser = argparse.ArgumentParser(description="대상 8종으로 시연용 가상 처방전을 만든다")
    parser.add_argument("--images", action="store_true", help="처방전 이미지(PNG)도 만든다")
    parser.add_argument("--font", help="한글 글꼴 파일 (기본: 운영체제의 기본 한글 글꼴)")
    parser.add_argument("--out", type=Path, default=SAMPLES, help="samples 폴더 (기본 agent/resources/samples)")
    args = parser.parse_args(argv)

    codes = {e.name: e.item_code for e in load_dictionary().entries}
    a1_dir, expected_dir, img_dir = args.out / "rx8_a1", args.out / "rx8_expected", args.out / "rx8_img"
    for folder in (a1_dir, expected_dir):
        folder.mkdir(parents=True, exist_ok=True)
    font = bold = None
    if args.images:
        found = next(((f, b) for f, b in DEFAULT_FONTS if Path(f).exists()), (None, None))
        font, bold = (args.font, None) if args.font else found
        if font is None:
            out("한글 글꼴을 찾지 못했습니다. --font 로 글꼴 파일을 알려 주세요.")
            return 1
        if bold and not Path(bold).exists():
            bold = None
    img_dir.mkdir(parents=True, exist_ok=True)
    detail, names = cv_ground_truth(codes)
    (img_dir / "ground_truth.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (img_dir / "gt.json").write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for number, rx in enumerate(PRESCRIPTIONS, start=1):
        a1 = a1_of(rx, number, codes)
        (a1_dir / f"{rx['id']}.json").write_text(json.dumps(a1, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (expected_dir / f"{rx['id']}.json").write_text(
            json.dumps(expected_of(rx, codes), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if font:
            draw(rx, number, codes, img_dir / f"{rx['id']}.png", font, bold)
        out(f"{rx['id']}: {' · '.join(d[0] for d in rx['drugs'])} ({rx['purpose']})")
    out(f"{len(PRESCRIPTIONS)}장을 만들었습니다: {a1_dir}" + (f" (이미지: {img_dir})" if font else ""))
    out(f"CV 정답: {img_dir / 'ground_truth.json'} · {img_dir / 'gt.json'}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
