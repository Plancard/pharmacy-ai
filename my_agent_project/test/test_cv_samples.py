"""CV 팀장이 보낸 실제 형식의 OCR 샘플(data/agent/cv_samples)과 우리 정답(data/agent/cv_expected)이
서로 맞는지 확인한다. 파서 테스트가 아니라 '데이터' 검사다.

- 샘플은 A.1 형식이다. 줄은 박스 중심 y 순서로 오지만 같은 행 안의 x 순서는 일정하지 않다.
- 약품 행은 '처방 의약품의 명칭' 표 안에서 9자리 코드가 든 줄이 기준이다.
- cv_expected는 'OCR 기준 정답'이다. OCR이 틀리게 읽은 글자까지 그대로 적고, 진짜 값은 true_* 칸에 둔다.
- cv_samples/ground_truth.json은 팀장이 준 진짜 값이다. cv_expected가 이것과 어긋나면 테스트가 알려 준다.
"""
import json
import re
from pathlib import Path

import pytest

from agent.drug_names import normalize_name, similarity

DATA = Path(__file__).resolve().parent.parent / "data" / "agent"
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]

NINE_DIGITS = re.compile(r"(?<!\d)\d{9}(?!\d)")
TABLE_HEADER_WORDS = ("횟수", "일수", "구분코드")          # 표 머리글의 마지막 줄들
DOSE = re.compile(r"^(\d+)\s*(TAB|CAP)$")
TIMES = re.compile(r"^(\d+)회$")
DAYS = re.compile(r"^(\d+)일$")
USAGE = "PO 식후 30분"
IN_TARGET_CODES = {"900000011", "900000021", "900000041", "900000051"}   # 타이레놀·게보린·트리스펜·프론크논
TARGET_NAMES = [   # 대상 약품 8종 (팀장 안내)
    "타이레놀정500mg", "게보린정", "트리스펜연질캡슐", "프론크논캡슐",
    "아르펜정", "네프신정", "트립라인정", "뮤코다인캡슐200mg",
]
# OCR이 발급번호를 틀리게 읽은 샘플: (OCR이 읽은 값, 진짜 값). 끝의 '호'를 '3'으로 읽었다.
KNOWN_ISSUE_NO_MISREADS = {"rx_004": ("2026년11월03일 000583", "2026년11월03일 00058호")}


def load(kind, sample_id):
    return json.loads((DATA / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def center_y(line):
    return (line["bbox"][1] + line["bbox"][3]) / 2


def table_band(ocr):
    """표 머리글 아래 ~ '주사제 처방명세' 위. 중심 y 순서."""
    lines = ocr["lines"]
    top = max(l["bbox"][3] for l in lines if l["text"] in TABLE_HEADER_WORDS)
    bottom = min(l["bbox"][1] for l in lines if l["text"].startswith("주사제 처방명세"))
    return sorted((l for l in lines if top < center_y(l) < bottom), key=center_y)


@pytest.fixture(params=SAMPLE_IDS)
def sample(request):
    sid = request.param
    return sid, load("cv_samples", sid), load("cv_expected", sid)


# ------------------------------------------------------------ 샘플 형식 (A.1)


def test_a1_format(sample):
    sid, ocr, _ = sample
    assert set(ocr) == {"image_id", "width", "height", "lines"}
    assert ocr["image_id"] == sid
    assert isinstance(ocr["width"], int) and isinstance(ocr["height"], int)
    assert ocr["lines"]
    for line in ocr["lines"]:
        assert set(line) == {"text", "bbox", "conf"}
        assert isinstance(line["text"], str) and line["text"].strip()
        x1, y1, x2, y2 = line["bbox"]
        assert 0 <= x1 < x2 <= ocr["width"] and 0 <= y1 < y2 <= ocr["height"], line
        assert 0.0 <= line["conf"] <= 1.0


def test_lines_come_in_center_y_order(sample):
    _, ocr, _ = sample
    ys = [center_y(l) for l in ocr["lines"]]
    assert ys == sorted(ys)


def test_cells_of_one_row_do_not_come_in_a_fixed_x_order():
    # 같은 행의 칸이 x 순서로 오지 않는 샘플이 실제로 있다. 파서가 목록 순서에 기대면 안 된다.
    ocr = load("cv_samples", "rx_001")
    first_row = [l for l in table_band(ocr) if center_y(l) < 1320]
    xs = [l["bbox"][0] for l in first_row]
    assert xs != sorted(xs)


# ------------------------------------------------------------ 정답의 모양


def test_expected_has_the_documented_shape(sample):
    sid, _, expected = sample
    assert set(expected) == {"image_id", "note", "issue_no", "true_issue_no", "drugs"}
    assert expected["image_id"] == sid
    assert expected["issue_no"].strip() and expected["drugs"]
    for drug in expected["drugs"]:
        assert set(drug) == {
            "printed_code", "name", "dose_per_time", "dose_unit", "times_per_day", "days",
            "usage_text", "true_name", "in_target", "item_code",
        }
        assert NINE_DIGITS.fullmatch(drug["printed_code"])
        assert drug["name"].strip()
        assert isinstance(drug["dose_per_time"], int) and drug["dose_per_time"] >= 1
        assert drug["dose_unit"] in ("TAB", "CAP")
        assert 1 <= drug["times_per_day"] <= 4 and drug["days"] >= 1
        assert isinstance(drug["in_target"], bool)


# ------------------------------------------------------------ 정답이 OCR 원문과 맞는지


def test_codes_in_the_table_match_the_expected_drugs_in_order(sample):
    _, ocr, expected = sample
    found = [m.group() for l in table_band(ocr) for m in [NINE_DIGITS.search(l["text"])] if m]
    assert found == [d["printed_code"] for d in expected["drugs"]]


def test_each_code_is_on_exactly_one_line_of_the_table(sample):
    _, ocr, expected = sample
    band_text = [l["text"] for l in table_band(ocr)]
    for drug in expected["drugs"]:
        assert sum(drug["printed_code"] in t for t in band_text) == 1


@pytest.mark.parametrize(
    "pattern, field",
    [(TIMES, "times_per_day"), (DAYS, "days")],
)
def test_times_and_days_per_row_match(sample, pattern, field):
    _, ocr, expected = sample
    got = [int(m.group(1)) for l in table_band(ocr) for m in [pattern.match(l["text"])] if m]
    assert got == [d[field] for d in expected["drugs"]]


def test_dose_and_unit_per_row_match(sample):
    _, ocr, expected = sample
    got = [(int(m.group(1)), m.group(2)) for l in table_band(ocr) for m in [DOSE.match(l["text"])] if m]
    assert got == [(d["dose_per_time"], d["dose_unit"]) for d in expected["drugs"]]


def test_usage_text_per_row_matches(sample):
    _, ocr, expected = sample
    got = [l["text"] for l in table_band(ocr) if l["text"] == USAGE]
    assert got == [d["usage_text"] for d in expected["drugs"]]


NAME_COLUMN_RIGHT_EDGE = 1250     # 이 샘플들에서 약품명 칸은 x 1250 왼쪽 (용량 칸은 1300부터)


def names_read_from_ocr(ocr):
    """OCR 원문에서 약품명을 정리해 읽는다 (파서와 따로 만든 단순한 검사용 구현).

    코드가 든 줄에서 앞 번호 (n)과 9자리 코드를 빼고 '/'를 공백으로 바꾸고,
    같은 행(y가 겹침)의 약품명 칸에 떨어져 나온 줄(예: 145mg)을 x 순서로 뒤에 붙인다.
    """
    band = table_band(ocr)
    anchors = [l for l in band if NINE_DIGITS.search(l["text"])]
    names = []
    for anchor in anchors:
        text = re.sub(r"^\(\d+\)", "", anchor["text"].strip())
        text = NINE_DIGITS.sub("", text).replace("/", " ")
        extras = [
            l for l in band
            if l is not anchor and l not in anchors and l["bbox"][2] < NAME_COLUMN_RIGHT_EDGE
            and l["bbox"][1] < anchor["bbox"][3] and l["bbox"][3] > anchor["bbox"][1]
        ]
        extras.sort(key=lambda l: l["bbox"][0])
        names.append(" ".join([text, *(l["text"] for l in extras)]).split())
    return [" ".join(words) for words in names]


def test_expected_names_are_the_ocr_text_cleaned_up(sample):
    _, ocr, expected = sample
    assert names_read_from_ocr(ocr) == [d["name"] for d in expected["drugs"]]


def test_issue_no_is_an_ocr_line(sample):
    _, ocr, expected = sample
    assert expected["issue_no"] in [l["text"] for l in ocr["lines"]]


# ------------------------------------------------------------ 약품 정보의 일관성 (샘플 전체)


def all_expected_drugs():
    return [d for sid in SAMPLE_IDS for d in load("cv_expected", sid)["drugs"]]


def test_a_code_always_means_the_same_drug_across_samples():
    seen = {}
    for d in all_expected_drugs():
        key = (d["true_name"], d["in_target"], d["item_code"])
        assert seen.setdefault(d["printed_code"], key) == key, d["printed_code"]


def test_in_target_drugs_have_their_printed_code_as_item_code():
    for d in all_expected_drugs():
        if d["in_target"]:
            assert d["item_code"] == d["printed_code"] and d["true_name"]
            assert d["printed_code"] in IN_TARGET_CODES
        else:
            assert d["item_code"] is None
            assert d["printed_code"] not in IN_TARGET_CODES


def test_sample_inventory_covers_the_cases_we_want_to_test():
    drugs = {sid: load("cv_expected", sid)["drugs"] for sid in SAMPLE_IDS}
    assert any(d["days"] > 3 for d in drugs["rx_003"])                       # 3일 트레이를 넘는 처방
    assert not any(d["in_target"] for d in drugs["rx_005"])                  # 전부 대상 외 실제 약품
    assert {d["printed_code"] for sid in SAMPLE_IDS for d in drugs[sid] if d["in_target"]} == IN_TARGET_CODES
    # 저신뢰(conf 0.5 미만) 약품명 줄의 수. 약사 확인으로 넘어가야 하는 줄을 시험하는 데 쓴다.
    low_conf_name_lines = {
        sid: sum(
            1 for l in table_band(load("cv_samples", sid)) if NINE_DIGITS.search(l["text"]) and l["conf"] < 0.5
        )
        for sid in SAMPLE_IDS
    }
    assert low_conf_name_lines == {"rx_001": 1, "rx_002": 0, "rx_003": 1, "rx_004": 3, "rx_005": 3}


# ------------------------------------------------------------ 팀장의 ground_truth.json과 대조


def ground_truth():
    raw = json.loads((DATA / "cv_samples" / "ground_truth.json").read_text(encoding="utf-8"))
    return {key.removesuffix(".png"): value for key, value in raw.items() if not key.startswith("_")}


def test_ground_truth_covers_exactly_these_samples():
    assert sorted(ground_truth()) == SAMPLE_IDS


def test_expected_agrees_with_the_ground_truth(sample):
    sid, _, expected = sample
    truth = ground_truth()[sid]
    assert expected["true_issue_no"] == truth["issue"]
    assert len(expected["drugs"]) == len(truth["drugs"])
    for mine, theirs in zip(expected["drugs"], truth["drugs"]):
        assert (
            mine["printed_code"], mine["dose_per_time"], mine["dose_unit"],
            mine["times_per_day"], mine["days"], mine["usage_text"], mine["true_name"],
        ) == (
            theirs["code"], theirs["dose"], theirs["unit"],
            theirs["times"], theirs["days"], theirs["usage"], theirs["name"],
        )


def test_ocr_issue_no_differs_from_the_truth_only_where_ocr_misread_it(sample):
    sid, _, expected = sample
    if sid in KNOWN_ISSUE_NO_MISREADS:
        assert (expected["issue_no"], expected["true_issue_no"]) == KNOWN_ISSUE_NO_MISREADS[sid]
    else:
        assert expected["issue_no"] == expected["true_issue_no"]


def test_in_target_flag_agrees_with_the_target_list():
    target_keys = {normalize_name(name) for name in TARGET_NAMES}
    for drug in all_expected_drugs():
        assert (normalize_name(drug["true_name"]) in target_keys) == drug["in_target"], drug["true_name"]


def test_ocr_names_still_look_like_the_true_names(sample):
    # 이름이 다른 약과 짝지어지는 실수를 막는다. 가장 많이 틀리게 읽은 이름도 이 정도는 비슷하다.
    _, _, expected = sample
    for drug in expected["drugs"]:
        assert similarity(normalize_name(drug["name"]), normalize_name(drug["true_name"])) >= 0.6, drug["name"]


def test_ocr_misreads_exist_in_the_samples():
    # 약품명·용량 글자를 틀리게 읽은 약품 수. 파서와 매칭을 시험하는 예로 쓴다.
    # (타이레놀 5oOmg 2개, 알모란정 4개, 프론크논 2개, 트리스펜 1개, rx_005 4개)
    misread = [d for d in all_expected_drugs() if normalize_name(d["name"]) != normalize_name(d["true_name"])]
    assert len(misread) == 13


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_samples_have_no_unmasked_resident_number(sample_id):
    # REQ-NF-02: 실제 환자 정보를 쓰지 않는다. 주민등록번호는 뒷자리가 가려져 있어야 한다 (900101-1******).
    texts = [line["text"] for line in load("cv_samples", sample_id)["lines"]]
    assert not any(re.search(r"\d{6}-?[1-4]\d{6}", t) for t in texts)
    assert any(re.search(r"\d{6}-[1-4]\*{6}", t) for t in texts)      # 가려진 칸은 있다 (검사가 헛돌지 않게)
