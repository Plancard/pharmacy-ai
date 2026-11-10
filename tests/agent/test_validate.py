import json
from pathlib import Path

import pytest

from agent.config import RESOURCES_DIR
from agent.errors import StructureError
from agent.structure import extract_rule
from agent.validate import default_times, order_from_extracted

DATA = RESOURCES_DIR / "samples"   # CV 샘플은 cv_a1/, 우리 정답은 cv_expected/ 에 동결해 둔다
SAMPLE_IDS = ["rx_001", "rx_002", "rx_003", "rx_004", "rx_005"]


def load(kind, sample_id):
    return json.loads((DATA / kind / f"{sample_id}.json").read_text(encoding="utf-8"))


def drug(**overrides):
    base = dict(
        name="알마겔정",
        printed_code="699000031",
        count_per_dose=1,
        times_per_day=3,
        days=3,
        times=["morning", "noon", "evening"],
    )
    base.update(overrides)
    return base


def build(*items, issue_no=None, source="rule", prescription_id="P0001"):
    extracted = {"issue_no": issue_no, "items": list(items) or [drug()]}
    return order_from_extracted(prescription_id, extracted, source=source)


# ------------------------------------------------------------ 규칙 경로 결과 -> 조제지시서


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_rule_output_becomes_the_expected_order(sample_id):
    extracted = extract_rule(load("cv_a1", sample_id))
    order = order_from_extracted("P0001", extracted, source="rule")
    expected = load("cv_expected", sample_id)
    assert order.issue_no == expected["issue_no"]
    assert order.source == "rule"
    got = [d.model_dump() for d in order.items]
    assert [{key: d[key] for key in ("name", "printed_code", "count_per_dose", "days", "times")} for d in got] == [
        {
            "name": want["name"],
            "printed_code": want["printed_code"],
            "count_per_dose": want["count_per_dose"],
            "days": want["days"],
            "times": default_times(want["times_per_day"]),   # 용법이 'PO 식후 30분'뿐이라 횟수로 채운다
        }
        for want in expected["items"]
    ]
    assert [d["ocr_conf"] for d in got] == [d["ocr_conf"] for d in extracted["items"]]      # 신뢰도는 그대로 넘어간다
    # 약품 매칭은 아직 하지 않았다
    assert all(d["item_code"] is None and not d["needs_pharmacist_check"] for d in got)
    # 약품마다 시점을 채웠다는 경고가 하나씩 남는다
    assert len(order.warnings) == len(expected["items"])
    assert all("처방전에 없어" in w for w in order.warnings)


@pytest.mark.parametrize("conf", [None, 0, 0.0, 0.43, 1, 1.0])
def test_ocr_conf_is_passed_to_the_line(conf):
    assert build(drug(ocr_conf=conf)).items[0].ocr_conf == conf


def test_a_drug_without_ocr_conf_is_fine():
    assert "ocr_conf" not in drug() and build(drug()).items[0].ocr_conf is None


@pytest.mark.parametrize("bad", ["0.5", True, False, [0.5], {"v": 0.5}])
def test_ocr_conf_of_the_wrong_type_is_refused(bad):
    with pytest.raises(StructureError, match="1번째 약품의 OCR 신뢰도"):
        build(drug(ocr_conf=bad))


@pytest.mark.parametrize("bad", [-0.1, 1.01, 5])
def test_ocr_conf_outside_zero_to_one_is_refused(bad):
    with pytest.raises(StructureError, match="1번째 약품의 값이 올바르지 않습니다"):
        build(drug(ocr_conf=bad))


def test_source_line_ids_are_passed_to_the_line():
    assert build(drug(source_line_ids=[17, 18, 20])).items[0].source_line_ids == [17, 18, 20]


def test_without_source_line_ids_the_list_is_empty():
    # LLM 경로는 원문의 어느 줄을 읽었는지 모른다
    assert build(drug()).items[0].source_line_ids == []
    assert build(drug(source_line_ids=None)).items[0].source_line_ids == []


def test_rule_output_keeps_the_source_line_ids():
    extracted = extract_rule(load("cv_a1", "rx_001"))
    order = order_from_extracted("P0001", extracted, source="rule")
    assert [d.source_line_ids for d in order.items] == [d["source_line_ids"] for d in extracted["items"]]
    assert all(d.source_line_ids for d in order.items)


@pytest.mark.parametrize("bad", ["3,4", 3, [-1], [1.0], ["3"], [True], {"a": 1}])
def test_bad_source_line_ids_are_refused(bad):
    with pytest.raises(StructureError, match="1번째 약품의 원문 줄 번호"):
        build(drug(source_line_ids=bad))


def test_item_code_is_left_empty_until_matching():
    assert build().items[0].item_code is None


def test_source_is_kept():
    assert build(source="llm").source == "llm"


def test_errors_module_is_shared():
    from agent.errors import StructureError as FromErrors
    from agent.structure import StructureError as FromStructure

    assert FromErrors is FromStructure


# ------------------------------------------------------------ 복용 시점 채우기


@pytest.mark.parametrize(
    "times, expected",
    [
        (1, ["morning"]),
        (2, ["morning", "evening"]),
        (3, ["morning", "noon", "evening"]),
        (4, ["morning", "noon", "evening", "bedtime"]),   # 트레이의 네 행 전부
    ],
)
def test_missing_times_are_filled_from_times_per_day_with_a_warning(times, expected):
    order = build(drug(times_per_day=times, times=[]))
    assert order.items[0].times == expected
    assert len(order.warnings) == 1
    assert "1번째 약품" in order.warnings[0] and "처방전에 없어" in order.warnings[0]


def test_times_none_is_treated_like_empty():
    order = build(drug(times=None))
    assert order.items[0].times == ["morning", "noon", "evening"] and order.warnings


def test_count_mismatch_is_refilled_with_a_warning_asking_for_review():
    order = build(drug(times_per_day=3, times=["morning"]))
    assert order.items[0].times == ["morning", "noon", "evening"]
    assert "확인이 필요" in order.warnings[0]


def test_explicit_times_are_kept_in_fixed_order_without_warning():
    order = build(drug(times_per_day=2, times=["evening", "morning"]))
    assert order.items[0].times == ["morning", "evening"]
    assert order.warnings == []


def test_bedtime_is_a_normal_time_slot_and_sorts_last():
    order = build(drug(times_per_day=2, times=["bedtime", "evening"]))
    assert order.items[0].times == ["evening", "bedtime"]
    assert order.warnings == []


def test_four_times_a_day_with_all_four_slots_is_accepted():
    order = build(drug(times_per_day=4, times=["bedtime", "evening", "noon", "morning"]))
    assert order.items[0].times == ["morning", "noon", "evening", "bedtime"]
    assert order.warnings == []


def test_default_slots_match_the_slot_order_for_every_allowed_count():
    from agent.schema import TIME_SLOTS
    from agent.validate import default_times

    for times in range(1, len(TIME_SLOTS) + 1):
        slots = default_times(times)
        assert len(slots) == times and set(slots) <= set(TIME_SLOTS)
        assert slots == [s for s in TIME_SLOTS if s in slots]   # 고정 순서


def test_duplicate_times_count_as_one_and_trigger_the_refill_warning():
    order = build(drug(times_per_day=2, times=["morning", "morning"]))
    assert order.items[0].times == ["morning", "evening"]
    assert "확인이 필요" in order.warnings[0]


def test_warning_names_which_drug():
    order = build(drug(), drug(name="게보린정", times=[]))
    assert len(order.warnings) == 1 and "2번째 약품" in order.warnings[0]


@pytest.mark.parametrize("bad", [["night"], ["morning", "sleep"], "morning", [["morning"]], 3])
def test_unknown_time_slot_values_fail_instead_of_being_dropped(bad):
    with pytest.raises(StructureError, match="복용 시점"):
        build(drug(times=bad))


# ------------------------------------------------------------ 숫자 검사


@pytest.mark.parametrize("field", ["count_per_dose", "times_per_day", "days"])
@pytest.mark.parametrize("bad", [1.0, "3", True, None])
def test_numbers_must_be_integers_and_are_never_converted(field, bad):
    with pytest.raises(StructureError, match="정수"):
        build(drug(**{field: bad}))


@pytest.mark.parametrize(
    "overrides, said",
    [
        ({"count_per_dose": 0}, "1회량"),
        ({"count_per_dose": -1}, "1회량"),
        ({"times_per_day": 0}, "복용 횟수"),
        ({"times_per_day": 5}, "복용 횟수"),   # 하루 최대 4회
        ({"days": 0}, "투약 일수"),
        ({"days": -3}, "투약 일수"),
    ],
)
def test_out_of_range_numbers_fail_and_say_which(overrides, said):
    with pytest.raises(StructureError, match=said):
        build(drug(**overrides))


def test_missing_number_field_fails():
    broken = drug()
    del broken["days"]
    with pytest.raises(StructureError, match="투약 일수"):
        build(broken)


# ------------------------------------------------------------ 약품·처방 단위 검사


@pytest.mark.parametrize("name", ["", "   ", None, 123])
def test_blank_or_non_text_name_fails(name):
    with pytest.raises(StructureError, match="약품명"):
        build(drug(name=name))


def test_drug_entry_must_be_an_object():
    with pytest.raises(StructureError, match="1번째 약품"):
        order_from_extracted("P0001", {"items": ["알마겔정"]}, source="rule")


@pytest.mark.parametrize("bad", [None, [], "텍스트", 3])
def test_extracted_must_be_an_object(bad):
    with pytest.raises(StructureError):
        order_from_extracted("P0001", bad, source="rule")


@pytest.mark.parametrize("items", [None, [], "알마겔정", {}])
def test_drugs_must_be_a_non_empty_list(items):
    with pytest.raises(StructureError, match="약품을 찾지 못했습니다"):
        order_from_extracted("P0001", {"items": items}, source="rule")


def test_missing_drugs_key_fails():
    with pytest.raises(StructureError):
        order_from_extracted("P0001", {"issue_no": "1"}, source="rule")


@pytest.mark.parametrize("prescription_id", ["", "  ", None, 1])
def test_prescription_id_is_required(prescription_id):
    with pytest.raises(StructureError, match="처방 ID"):
        build(prescription_id=prescription_id)


def test_second_drug_error_is_numbered():
    with pytest.raises(StructureError, match="2번째"):
        build(drug(), drug(days=0))


# ------------------------------------------------------------ 글자 다듬기


def test_prescription_id_is_stripped():
    assert build(prescription_id=" P0001 ").prescription_id == "P0001"


def test_text_is_stripped_and_blank_optional_fields_become_none():
    order = build(drug(name="  알마겔정 ", printed_code="   "), issue_no="")
    assert order.items[0].name == "알마겔정"
    assert order.items[0].printed_code is None
    assert order.issue_no is None


def test_issue_no_and_code_are_kept_when_present():
    order = build(drug(printed_code=" 699000031 "), issue_no=" 2026110400001 ")
    assert order.items[0].printed_code == "699000031"
    assert order.issue_no == "2026110400001"


def test_non_text_code_or_issue_no_fails():
    with pytest.raises(StructureError, match="코드"):
        build(drug(printed_code=699000031))
    with pytest.raises(StructureError, match="교부번호"):
        build(issue_no=2026110400001)


def test_unknown_extra_keys_are_ignored():
    order = build(drug(memo="무시됨"))
    assert order.items[0].name == "알마겔정"


def test_error_messages_do_not_contain_prescription_text():
    with pytest.raises(StructureError) as error:
        build(drug(name="비밀약품", count_per_dose=0))
    assert "비밀약품" not in str(error.value)
