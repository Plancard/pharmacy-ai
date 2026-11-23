"""대상 8종 시연용 가상 처방전 (agent/resources/samples/rx8_*, scripts/agent/make_prescriptions.py).

처방전마다 구조화 → 약품 매칭 → 위험 신호 대조 → (승인) → 조제 계획이 정답(rx8_expected)과 같은지 본다.
"""
import importlib.util
import json
from pathlib import Path

import pytest

import fake_langgraph
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import load_dictionary
from agent.drug_match import match_order
from agent.errors import E_PLAN_RANGE
from agent.checkpoint import memory_checkpointer
from agent.graph import build_graph, resume_run
from agent.pipeline import structure_prescription
from agent.plan import PlanError, build_plan
from agent.risk import NO_DUR_WARNING, UNCHECKED_TAIL, check_order
from agent.schema import TRAY_DAYS, OcrResult
from graph_calls import first_state, send

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = RESOURCES_DIR / "samples"
SCRIPT = ROOT / "scripts" / "agent" / "make_prescriptions.py"
_spec = importlib.util.spec_from_file_location("make_prescriptions", SCRIPT)
make_prescriptions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_prescriptions)

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
IDS = [rx["id"] for rx in make_prescriptions.PRESCRIPTIONS]


def load(kind, rx_id):
    return json.loads((SAMPLES / kind / f"{rx_id}.json").read_text(encoding="utf-8"))


def no_llm(text, feedback):
    raise AssertionError("시연용 처방전은 규칙 경로로 읽혀야 한다")


def checked(rx_id):
    order = structure_prescription(f"P-{rx_id}", load("rx8_a1", rx_id), llm_fn=no_llm)
    return check_order(match_order(order, DICT, api=None), DICT)


def test_there_are_twenty_prescriptions_with_files():
    """REQ-AG-02 · 03 수용 기준의 가상 처방전 20장."""
    assert IDS == [f"rx8_{i:02d}" for i in range(1, 21)]
    for rx_id in IDS:
        assert (SAMPLES / "rx8_a1" / f"{rx_id}.json").exists() and (SAMPLES / "rx8_expected" / f"{rx_id}.json").exists()


def test_the_files_are_what_the_script_makes(tmp_path):
    """표(PRESCRIPTIONS)를 고치고 스크립트를 다시 돌리지 않으면 여기서 알린다."""
    make_prescriptions.main(["--out", str(tmp_path)], out=lambda line: None)
    for kind in ("rx8_a1", "rx8_expected"):
        for rx_id in IDS:
            assert (tmp_path / kind / f"{rx_id}.json").read_text(encoding="utf-8") == (SAMPLES / kind / f"{rx_id}.json").read_text(encoding="utf-8")


def test_every_target_drug_appears_and_nothing_else():
    used = {item["item_code"] for rx_id in IDS for item in load("rx8_expected", rx_id)["items"] if item["target"]}
    assert used == {e.item_code for e in DICT.entries}


@pytest.mark.parametrize("rx_id", IDS)
def test_the_ocr_is_a_valid_a1_in_the_cv_form_with_only_virtual_people(rx_id):
    """CV 팀 가상 처방전과 같은 양식이다. 환자 · 의료인은 가상이고 주민등록번호 뒷자리는 가려져 있다."""
    ocr = OcrResult.model_validate(load("rx8_a1", rx_id))
    texts = [line.text for line in ocr.lines]
    cv_form = {line["text"] for line in load("cv_a1", "rx_001")["lines"]}
    for label in ("[별지 제9호 서식]", "처방 의약품의 명칭", "투약량", "횟수", "일수", "용", "법", "조제 시", "주민등록번호", "면허종류"):
        assert label in texts and label in cv_form
    rx = next(r for r in make_prescriptions.PRESCRIPTIONS if r["id"] == rx_id)
    assert any(t.startswith(rx["patient"] + "(") for t in texts)
    assert "900101-1******" in texts and "가상" in rx["hospital"]
    assert "※ OCR 테스트용 가상 처방전" in texts
    name, *_ = rx["drugs"][0]
    code = next(e.item_code for e in DICT.entries if e.name == name)
    rows = [t for t in texts if t.startswith("(1)")]                    # CV 양식: 첫 약품 줄만 앞에 (1), 이름은 이름/함량
    assert rows == [f"(1){code} {make_prescriptions.shown_name(name)}"]


@pytest.mark.parametrize("rx_id", IDS)
def test_each_prescription_reads_matches_and_checks_as_expected(rx_id):
    expected = load("rx8_expected", rx_id)
    order = checked(rx_id)
    assert order.source == "rule"
    keys = ("item_code", "name", "count_per_dose", "times", "days")
    squash = lambda text: "".join(text.split())      # 표준 이름은 공백이 없고 처방전 글자는 있다
    got = [{k: squash(d.printed_name or d.name) if k == "name" else getattr(d, k) for k in keys} for d in order.items]    # 이름은 처방전에서 읽은 글자
    assert got == [{k: squash(item[k]) if k == "name" else item[k] for k in keys} for item in expected["items"]]
    # 코드로 정한 대상 약은 표준 이름으로 보이고, 처방전 글자가 다르면 printed_name에 남는다
    for d, item in zip(order.items, expected["items"]):
        if item["target"]:
            assert d.name == DICT.by_item_code(item["item_code"]).name
            assert d.printed_name in (None, item["name"]) and (d.printed_name is None) == (squash(d.name) == squash(item["name"]))
    # 대상 8종은 인쇄된 품목기준코드로 자동 확정, 대상 외 약품은 정하지 못하고 약사 확인 (REQ-AG-03)
    assert [d.match_method for d in order.items] == ["code" if item["target"] else "none" for item in expected["items"]]
    assert order.flagged_indices == expected["flagged"]
    # 용법이 "PO 식후 30분"인 줄은 시점이 없어 하루 횟수로 배정했다는 경고가 붙는다 (시점을 적은 줄은 붙지 않는다)
    after_meal = [(i, len(item["times"])) for i, item in enumerate(expected["items"], start=1) if item["usage"] == "PO 식후 30분"]
    unmatched = [f"{item['name']}: 품목기준코드를 정하지 못해 {UNCHECKED_TAIL}" for item in expected["items"] if not item["target"]]
    assert order.warnings == [f"{i}번째 약품: 복용 시점이 처방전에 없어 하루 {n}회 기준으로 배정했습니다." for i, n in after_meal] \
        + unmatched + [NO_DUR_WARNING]                                 # 시험은 DUR API 없이 돈다
    assert order.risk_check.level == expected["risk_level"]


@pytest.mark.parametrize("rx_id", IDS)
def test_each_prescription_plans_as_expected(rx_id):
    expected = load("rx8_expected", rx_id)
    order = checked(rx_id)
    if expected["plan"] == "ok":
        plan = build_plan(order)
        assert plan.days <= TRAY_DAYS
    else:
        with pytest.raises(PlanError) as error:
            build_plan(order)
        assert error.value.code == {"E_PLAN_RANGE": E_PLAN_RANGE, "unmatched": None}[expected["plan"]]


def test_the_demo_combinations_are_the_two_warnings_that_can_be_planned():
    warnings = [rx_id for rx_id in IDS if load("rx8_expected", rx_id)["risk_level"] == "warning" and load("rx8_expected", rx_id)["plan"] == "ok"]
    assert warnings == ["rx8_01", "rx8_02", "rx8_07"]                     # rx8_07: 조합 A + 뮤코다인 (한 칸 3알, 칸당 제한 없음)


@pytest.mark.parametrize("rx_id", IDS)
def test_each_prescription_stops_for_approval_and_then_plans(monkeypatch, rx_id):
    fake_langgraph.install(monkeypatch)
    graph = build_graph(dictionary=DICT, structure=lambda pid, ocr: structure_prescription(pid, ocr, llm_fn=no_llm), api=None,
                        checkpointer=memory_checkpointer())
    shown = send(graph, f"P-{rx_id}", load("rx8_a1", rx_id))["__interrupt__"][0].value
    expected = load("rx8_expected", rx_id)
    assert ("확인이 필요한 약품" in shown["notice"]) is bool(expected["flagged"])
    if expected["plan"] == "ok":                                      # 승인 카드에 조제 계획이 함께 있다 (v0.8)
        assert shown["notice"].startswith("약사 승인을 기다립니다.") and shown["plan"] is not None
        final = resume_run(graph, f"P-{rx_id}", {"action": "approve", "order": shown["order"]})
        assert final["status"] == "planned" and final["plan"] == shown["plan"]   # 보인 계획을 그대로 확정
    else:                                                             # 계획을 못 만들면 승인 카드에 이유를 보이고 계획은 없다
        assert shown["notice"].startswith("조제 계획을 만들 수 없습니다.") and shown["plan"] is None
        assert shown["error_code"] == {"E_PLAN_RANGE": E_PLAN_RANGE, "unmatched": None}[expected["plan"]]


def test_the_script_output_can_be_written_in_cp949(tmp_path):
    lines = []
    make_prescriptions.main(["--out", str(tmp_path)], out=lines.append)
    "\n".join(lines).encode("cp949")


def test_the_cv_ground_truth_lists_the_printed_names(tmp_path):
    """CV 팀이 cv/ocr_test.py --gt 로 OCR 약품명 인식률을 잴 수 있게, 같은 형식의 정답을 함께 둔다."""
    make_prescriptions.main(["--out", str(tmp_path)], out=lambda line: None)
    for name in ("ground_truth.json", "gt.json"):
        assert (tmp_path / "rx8_img" / name).read_text(encoding="utf-8") == (SAMPLES / "rx8_img" / name).read_text(encoding="utf-8")
    gt = json.loads((SAMPLES / "rx8_img" / "gt.json").read_text(encoding="utf-8"))
    detail = json.loads((SAMPLES / "rx8_img" / "ground_truth.json").read_text(encoding="utf-8"))
    assert list(gt) == [f"{rx_id}.png" for rx_id in IDS] == list(detail)
    assert gt["rx8_01.png"] == ["타이레놀정/500mg", "게보린정"]
    assert detail["rx8_05.png"]["drugs"][1] == {"code": "200002714", "name": "트립라인정", "dose": 1, "unit": "TAB",
                                                "times": 1, "days": 3, "usage": "PO 취침 전"}
    for image, names in gt.items():
        assert [d["name"] for d in detail[image]["drugs"]] == names


def test_the_images_exist_for_every_prescription():
    for rx_id in IDS:
        assert (SAMPLES / "rx8_img" / f"{rx_id}.png").stat().st_size > 10_000



def test_the_test_set_covers_the_acceptance_criteria():
    """20장 구성: 8종 전부 · 시연 중복 2쌍(경고) · 이부프로펜 처방 2장(아르펜 · 트리스펜) · 트레이 초과 1장 · 대상 외 약품 2장."""
    expected = {rx_id: load("rx8_expected", rx_id) for rx_id in IDS}
    plans = [e["plan"] for e in expected.values()]
    assert plans.count("ok") == 17 and plans.count("E_PLAN_RANGE") == 1 and plans.count("unmatched") == 2   # rx8_18 (4일)
    assert [rx_id for rx_id, e in expected.items() if not all(i["target"] for i in e["items"])] == ["rx8_19", "rx8_20"]
    ibuprofen = {"201110120", "200201239"}
    taboo_demo = [rx_id for rx_id in ("rx8_16", "rx8_17") if ibuprofen & {i["item_code"] for i in expected[rx_id]["items"]}]
    assert taboo_demo == ["rx8_16", "rx8_17"]
    counts = [len(e["items"]) for e in expected.values()]
    assert {1, 2, 3} <= set(counts)                                       # 약 1 · 2 · 3종 처방이 모두 있다
