import pytest
from pydantic import ValidationError

from agent.schema import DispensingOrder, DrugLine


def make(**overrides):
    base = dict(
        name="타이레놀정500밀리그램",
        dose_per_time=1,
        times_per_day=3,
        days=3,
        time_slots=["morning", "noon", "evening"],
    )
    base.update(overrides)
    return DrugLine(**base)


def make_order(**overrides):
    base = dict(prescription_id="P0001", source="rule", drugs=[make()])
    base.update(overrides)
    return DispensingOrder(**base)


# ------------------------------------------------------------ DrugLine


def test_valid_line():
    assert make().times_per_day == 3


def test_slot_values_are_the_four_tray_rows_in_order():
    from agent.schema import SLOT_ORDER

    assert SLOT_ORDER == ("morning", "noon", "evening", "bedtime")


def test_four_times_a_day_uses_all_four_rows():
    line = make(times_per_day=4, time_slots=["morning", "noon", "evening", "bedtime"])
    assert line.time_slots[-1] == "bedtime"


@pytest.mark.parametrize(
    "bad",
    [
        {"dose_per_time": 0},
        {"times_per_day": 4},
        {"days": 0},
        {"time_slots": ["morning"]},
        {"time_slots": ["morning", "morning", "noon"]},
        {"times_per_day": 4, "time_slots": ["morning", "noon", "evening", "night"]},   # 모르는 시점 이름
        {"times_per_day": 5, "time_slots": ["morning", "noon", "evening", "bedtime"]},   # 하루 최대 4회
        {"item_code": "123"},
        {"item_code": "12345678a"},
        {"name": "  "},
    ],
)
def test_invalid_line(bad):
    with pytest.raises(ValidationError):
        make(**bad)


# ------------------------------------------------------------ DispensingOrder


def test_valid_order():
    order = make_order(issue_no="2026110400001")
    assert order.prescription_id == "P0001" and order.warnings == []


def test_order_survives_dump_and_load():
    order = make_order()
    assert DispensingOrder.model_validate(order.model_dump(mode="json")) == order


@pytest.mark.parametrize(
    "bad",
    [
        {"drugs": []},
        {"prescription_id": "  "},
        {"source": "manual"},
    ],
)
def test_invalid_order(bad):
    with pytest.raises(ValidationError):
        make_order(**bad)


def test_all_matched_needs_item_code_on_every_line():
    assert make_order().all_matched is False
    both = [make(item_code="900000011"), make(item_code="900000021")]
    assert make_order(drugs=both).all_matched is True
    assert make_order(drugs=[make(item_code="900000011"), make()]).all_matched is False


def test_warnings_are_not_shared_between_orders():
    a, b = make_order(), make_order()
    a.warnings.append("확인 필요")
    assert b.warnings == []


# ------------------------------------------------------------ 약품 매칭 필드 (11/5-③)


def cand(**overrides):
    from agent.schema import MatchCandidate

    base = dict(item_code="900000021", name="게보린정", score=0.82, reason="이름이 비슷함")
    base.update(overrides)
    return MatchCandidate(**base)


def test_tray_days_is_three():
    from agent.schema import TRAY_DAYS

    assert TRAY_DAYS == 3


def test_a_new_line_is_unmatched_and_unflagged():
    line = make()
    assert (line.item_code, line.match_method, line.match_candidates) == (None, "none", [])
    assert line.needs_pharmacist_check is False and line.check_reasons == [] and line.ocr_conf is None


@pytest.mark.parametrize("code", ["900000011", "000000000"])
def test_item_code_of_nine_digits_is_accepted(code):
    assert make(item_code=code).item_code == code


@pytest.mark.parametrize("code", ["90000001", "9000000110", "90000001a", "９００００００１１", ""])
def test_item_code_must_be_nine_ascii_digits(code):
    with pytest.raises(ValidationError):
        make(item_code=code)


@pytest.mark.parametrize("conf", [-0.01, 1.01])
def test_ocr_conf_is_between_zero_and_one(conf):
    with pytest.raises(ValidationError):
        make(ocr_conf=conf)
    assert make(ocr_conf=0).ocr_conf == 0 and make(ocr_conf=1).ocr_conf == 1


@pytest.mark.parametrize("method", ["code", "name"])
def test_a_match_method_needs_an_item_code(method):
    with pytest.raises(ValidationError, match="item_code가 있어야"):
        make(match_method=method)
    assert make(match_method=method, item_code="900000021").match_method == method


def test_unknown_match_method_is_refused():
    with pytest.raises(ValidationError):
        make(match_method="guess", item_code="900000021")


def test_reasons_need_the_check_flag():
    with pytest.raises(ValidationError, match="needs_pharmacist_check"):
        make(check_reasons=["이유"])
    assert make(check_reasons=["이유"], needs_pharmacist_check=True).check_reasons == ["이유"]
    assert make(needs_pharmacist_check=True).check_reasons == []     # 표시만 있고 이유가 없어도 된다


def test_a_candidate():
    c = cand()
    assert (c.item_code, c.name, c.score, c.reason) == ("900000021", "게보린정", 0.82, "이름이 비슷함")
    assert make(match_candidates=[c]).match_candidates == [c]


@pytest.mark.parametrize("bad", [{"item_code": "9000000"}, {"item_code": "９００００００２１"}, {"score": 1.01}, {"score": -0.1}])
def test_a_bad_candidate_is_refused(bad):
    with pytest.raises(ValidationError):
        cand(**bad)


def test_candidate_lists_are_not_shared_between_lines():
    a, b = make(), make()
    a.match_candidates.append(cand())
    a.check_reasons.append("x")
    assert b.match_candidates == [] and b.check_reasons == []


def test_flagged_indices():
    flagged = make(needs_pharmacist_check=True)
    order = make_order(drugs=[make(), flagged, make(), flagged])
    assert order.flagged_indices == [1, 3]
    assert make_order().flagged_indices == []


def test_match_fields_survive_dump_and_load():
    line = make(item_code="900000021", match_method="code", match_candidates=[cand()],
                needs_pharmacist_check=True, check_reasons=["이유"], ocr_conf=0.43)
    order = make_order(drugs=[line])
    assert DispensingOrder.model_validate_json(order.model_dump_json()) == order
