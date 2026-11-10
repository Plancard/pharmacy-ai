"""다른 파트가 실제로 쓰는 값이 schema.py 형식을 지키는지 검사한다 (계약 검사).

- CV가 올려 둔 A.1 샘플 (cv/fixtures/a1/*.json)과 CV 가짜 모듈(cv/stub.py)의 A.1 · A.3 결과
- 웹의 가짜 Agent 연결부(web/agent_gateway.py)가 만드는 A.5

대상 파일이 없거나 Flask가 없으면 그 검사만 건너뛴다. 여기서 깨지면 schema.py나 상대 파트 둘 중 하나가 부록 A와 어긋난 것이다.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.schema import DispensingOrder, Inspection, OcrResult, Plan

ROOT = Path(__file__).resolve().parents[2]
CV_A1 = sorted((ROOT / "cv" / "fixtures" / "a1").glob("*.json"))


@pytest.mark.skipif(not CV_A1, reason="cv/fixtures/a1 이 없다")
@pytest.mark.parametrize("path", CV_A1, ids=lambda p: p.stem)
def test_cv_fixture_is_a1(path):
    OcrResult.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.fixture
def stub():
    return pytest.importorskip("cv.stub")


def sample_plan() -> dict:
    from agent.config import RESOURCES_DIR
    return json.loads((RESOURCES_DIR / "samples" / "a4_plan" / "s01_combo_a.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(not CV_A1, reason="cv/fixtures/a1 이 없다")
@pytest.mark.parametrize("path", CV_A1, ids=lambda p: p.stem)
def test_frozen_copy_still_equals_the_cv_original(path):
    """우리 시험은 cv/fixtures/a1을 복사해 동결한 agent/resources/samples/cv_a1을 쓴다. CV가 원본을 바꾸면 여기서 알린다."""
    from agent.config import RESOURCES_DIR
    frozen = RESOURCES_DIR / "samples" / "cv_a1" / path.name
    assert frozen.exists(), f"{path.name}: cv_a1/ 에 복사본이 없습니다 (새 샘플이면 복사하고 cv_expected/ 에 정답도 만드세요)"
    assert json.loads(frozen.read_text(encoding="utf-8")) == json.loads(path.read_text(encoding="utf-8")), \
        f"CV가 {path.name}을 바꿨습니다: 복사본과 cv_expected/ 정답을 함께 갱신하세요"


def test_cv_stub_ocr_is_a1(stub):
    if not CV_A1:
        pytest.skip("cv/fixtures/a1 이 없다")
    OcrResult.model_validate(stub.ocr(f"{CV_A1[0].stem}.png"))


@pytest.mark.parametrize("scenario", ["ok", "mismatch", "review", "rotated"])
def test_cv_stub_inspection_is_a3(stub, scenario):
    plan = Plan.model_validate(sample_plan())                      # 우리가 보내는 A.4
    res = stub.inspect("tray_demo.jpg", plan.model_dump(mode="json"), scenario)
    insp = Inspection.model_validate(res)
    assert insp.prescription_id == plan.prescription_id


def test_plan_range_is_rejected_by_both_sides(stub):
    """트레이 범위를 넘은 계획은 Agent 쪽 형식도, CV도 받지 않는다 (E_PLAN_RANGE)."""
    bad = {"prescription_id": "P0001", "days": 3,
           "plan": [{"day": 4, "time": "morning", "items": [{"item_code": "202106092", "count": 1}]}]}
    with pytest.raises(ValidationError):
        Plan.model_validate(bad)
    with pytest.raises(stub.PlanRangeError) as e:
        stub.inspect("tray_demo.jpg", bad)
    assert e.value.code == "E_PLAN_RANGE"


@pytest.mark.parametrize("scenario", ["warning", "none", "danger", "flagged"])
def test_web_fake_order_is_a5(scenario, tmp_path):
    pytest.importorskip("flask")
    from web.agent_gateway import FakeGateway
    order = FakeGateway(tmp_path, scenario)._order("P0001")
    parsed = DispensingOrder.model_validate(order)
    assert parsed.risk_check is not None
    assert parsed.risk_check.level == {"warning": "warning", "none": "none", "danger": "danger", "flagged": "warning"}[scenario]


CLASSES = ROOT / "cv" / "classes.yaml"


@pytest.mark.skipif(not CLASSES.exists(), reason="cv/classes.yaml 이 없다")
def test_target_drug_names_equal_cv_classes():
    """대상 약 8종의 이름·순서는 CV의 classes.yaml(확정 알약)과 같다. 어긋나면 우리 이름표나 CV 쪽을 먼저 맞춘다."""
    import re
    from test_schema import TARGET_8                    # 우리가 쓰는 8종 (품목기준코드 → 이름)

    rows = re.findall(r"\{id: (\d+), k_code: (K-\d+), name: (.+?)\}", CLASSES.read_text(encoding="utf-8"))
    assert [int(r[0]) for r in rows] == list(range(8))
    assert [r[2] for r in rows] == list(TARGET_8.values())
