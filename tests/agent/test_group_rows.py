import json
import random
from pathlib import Path

import pytest

from agent.config import RESOURCES_DIR
from agent.structure import group_rows, row_text

DATA = RESOURCES_DIR / "samples"   # CV 샘플은 cv_a1/, 우리 정답은 cv_expected/ 에 동결해 둔다

def load_lines(sample_id):
    path = DATA / "cv_a1" / f"{sample_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))["lines"]


# rx_001을 줄로 묶은 결과 전체. OCR 원문을 눈으로 보고 확인했다 (OCR이 틀리게 읽은 글자도 그대로).
RX_001_ROWS = [
    "[별지 제9호 서식]",
    "가정의학과 처   방   전 약국제출용",
    "건강보험 의로급여 산업재해보험 자동차보험 [기타 요양기관기호 99990001",
    "발급 연월일 및 번호 2026년11월03일 00012호 명   칭 (의)가상의료재단 테스트병원",
    "성   명 김가상(000000101) 잎 전화번호 02-000-0000",
    "환",
    "기",
    "팩스번호 02-000-0001",
    "자 주민등록번호 900101-1****** 관",
    "흉페이지 WWWtest- -hosp.example",
    "류출 R51 처 방 면허종류 의사",
    "의료인의 이테스트",
    "성 면히번호 제 000O1",
    "%환자의 요구가 있을 때에는 질병분류기호름 기재하지 아니합니다 .",
    "1회 1일 본인",
    "처방 의약품의 명칭 투약량 두여 투약 부담률 용 법",
    "횟수 일수 구분코드",
    "(1)900000011 타이레놀정/ 5oOmg 1 TAB 3회 3일 PO 식후 30분",
    "900000031 알모란정 1 TAB 3회 3일 PO 식후 30분",
    "주사제 처방명세([  ]원 내 조제, J원 외 처방) 조제 시 본인부담",
    "참고 사항 구분기호",
    "사용기간 발급일부터( )일간 사용기간내에 약국에 제출하여야 합니다.",
    "출력시간 : 2026.11.03 09.12.40",
    "의약품 조제명 세",
    "조제 기관의 명칭",
    "처방의 변경 . 수정 . 확인 . 대체시 그 내용 등",
    "조제약사 성명",
    "% OCR 테스트용 가상 처방전 실제 환자 . 의료기관 . 의료인 정보가 아입니다",
]

# 약품 줄: 코드 줄·떨어져 나온 함량·값 칸·용법이 왼쪽부터 한 줄로 모인다
DRUG_ROWS = {
    "rx_002": [
        "(1)900000021 게보린정 1 TAB 2회 2일 PO 식후 30분",
        "900000051 프론크논캠술 1 CAP 3회 2일 PO 식후 30분",
        "900000031 알모란정 1 TAB 3회 2일 PO 식후 30분",
    ],
    "rx_005": [
        "(1)671805860 필루비서방정 145mg 1 TAB 2회 3일 PO 식후 30분",
        "671807390 에스코렌정 / 1Omg 1 TAB 2회 3일 PO 식후 30분",
        "644500700 킬렉시아정 /25Omg 1 TAB 2회 3일 PO 식후 30분",
        "642202200 타세늘8시간이말서방정 / 65Omg 1 TAB 2회 3일 PO 식후 30분",
    ],
}


def test_rows_of_a_real_sample_are_grouped_top_to_bottom_and_left_to_right():
    assert [row_text(r) for r in group_rows(load_lines("rx_001"))] == RX_001_ROWS


@pytest.mark.parametrize("sample_id", sorted(DRUG_ROWS))
def test_each_drug_row_becomes_one_line(sample_id):
    rows = [row_text(r) for r in group_rows(load_lines(sample_id))]
    start = rows.index("횟수 일수 구분코드") + 1
    assert rows[start:start + len(DRUG_ROWS[sample_id])] == DRUG_ROWS[sample_id]
    assert rows[start + len(DRUG_ROWS[sample_id])].startswith("주사제 처방명세")


@pytest.mark.parametrize("sample_id", ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"])
@pytest.mark.parametrize("seed", range(3))
def test_input_order_does_not_matter(sample_id, seed):
    lines = load_lines(sample_id)
    shuffled = lines[:]
    random.Random(seed).shuffle(shuffled)
    assert [row_text(r) for r in group_rows(shuffled)] == [row_text(r) for r in group_rows(lines)]


def test_empty_input():
    assert group_rows([]) == []


def test_boxes_with_slightly_different_y_are_one_row_but_far_ones_are_not():
    lines = [
        {"text": "B", "bbox": [200, 104, 230, 134], "conf": 0.9},
        {"text": "A", "bbox": [100, 100, 130, 130], "conf": 0.9},
        {"text": "C", "bbox": [100, 160, 130, 190], "conf": 0.9},
    ]
    assert [row_text(r) for r in group_rows(lines)] == ["A B", "C"]


def _box(text, x1, y1, y2):
    return {"text": text, "bbox": [x1, y1, x1 + 50, y2], "conf": 0.9}


@pytest.mark.parametrize(
    "second_y1, together",
    [
        (120, True),     # 중심 차이 20 = 칸 높이(40)의 절반: 같은 줄
        (121, False),    # 21: 다른 줄
        (118, True),     # 18: 높이의 1/3(13.3)보다 크지만 절반 안이다. 같은 줄
        (100, True),     # 0
    ],
)
def test_same_row_means_center_distance_within_half_the_box_height(second_y1, together):
    first = _box("A", 0, 100, 140)                            # 중심 120
    second = _box("B", 100, second_y1, second_y1 + 40)         # 높이 40
    rows = [row_text(r) for r in group_rows([first, second])]
    assert rows == (["A B"] if together else ["A", "B"])


def test_boxes_with_the_same_center_give_the_same_rows_in_any_input_order():
    # 실제 rx_001에서 나온 모양: 키가 큰 '류출'과 '의료인의'의 세로 중심이 909.5로 같다.
    # 어느 쪽을 먼저 처리하느냐에 따라 줄 평균이 달라져, 예전에는 입력 순서에 따라 묶음이 바뀌었다.
    r51 = {"text": "R51", "bbox": [467, 853, 549, 900], "conf": 0.9}
    tall = {"text": "류출", "bbox": [266, 839, 345, 980], "conf": 0.9}
    small = {"text": "의료인의", "bbox": [991, 890, 1109, 929], "conf": 0.9}
    results = {
        tuple(row_text(r) for r in group_rows(order))
        for order in ([r51, tall, small], [r51, small, tall], [small, tall, r51], [tall, small, r51])
    }
    # 예전 방식은 ('류출 R51 의료인의',)와 ('R51', '류출 의료인의') 두 가지가 나왔다. 이제는 한 가지다.
    assert results == {("류출 R51 의료인의",)}
