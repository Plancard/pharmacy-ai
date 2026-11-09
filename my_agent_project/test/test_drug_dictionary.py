"""약품 사전 읽기 (agent/drug_dictionary.py)."""
import json

import pytest

from agent.config import DATA_DIR, DRUG_DICTIONARY_PATH
from agent.drug_dictionary import dictionary_from_data, load_dictionary
from agent.errors import ConfigError

GROUND_TRUTH = json.loads((DATA_DIR / "cv_samples" / "ground_truth.json").read_text(encoding="utf-8"))


def item(code="900000011", name="타이레놀정 500mg"):
    return {"item_code": code, "name": name}


# ------------------------------------------------------------ 실제 사전 파일


def test_the_default_path_is_the_project_dictionary():
    assert DRUG_DICTIONARY_PATH == DATA_DIR / "drug_dictionary.json"
    assert DATA_DIR.name == "agent" and DATA_DIR.parent.name == "data"


# 명세서 v0.3 2.6절 대상 약품 표: (클래스 ID, 품목기준코드, 제품명)
SPEC_2_6 = [
    (0, "202106092", "타이레놀정 500mg"),
    (1, "197900277", "게보린정"),
    (2, "201110120", "트리스펜연질캡슐"),
    (3, "200200099", "프론크논캡슐"),
    (4, "200201239", "아르펜정"),
    (5, "201405845", "네프신정"),
    (6, "200002714", "트립라인정"),
    (7, "200308774", "뮤코다인캡슐 200mg"),
]


def test_the_project_dictionary_is_the_eight_drugs_of_the_spec():
    dictionary = load_dictionary()
    assert [(e.class_id, e.item_code, e.name) for e in dictionary.entries] == SPEC_2_6
    assert len(dictionary) == 8


def test_prescription_codes_of_the_project_dictionary():
    # 가상 처방전에 인쇄되는 코드. 처방전에 나오지 않는 4종은 없다
    assert {e.name: e.rx_code for e in load_dictionary().entries} == {
        "타이레놀정 500mg": "900000011", "게보린정": "900000021", "트리스펜연질캡슐": "900000041", "프론크논캡슐": "900000051",
        "아르펜정": None, "네프신정": None, "트립라인정": None, "뮤코다인캡슐 200mg": None,
    }


def test_dictionary_rx_codes_agree_with_the_ground_truth():
    # 정답 파일에 같은 코드로 나오는 약은 사전과 같은 약이어야 한다 ('/'와 띄어쓰기만 다르다)
    prescriptions = [rx for key, rx in GROUND_TRUTH.items() if key.endswith(".png")]
    assert len(prescriptions) == 5
    truth = {d["code"]: d["name"] for rx in prescriptions for d in rx["drugs"]}
    entries = [e for e in load_dictionary().entries if e.rx_code]
    assert len(entries) == 4
    for entry in entries:
        assert truth[entry.rx_code].replace("/", " ") == entry.name


def test_class_ids_agree_with_the_cv_classes_file():
    # CV의 classes.yaml과 같은 클래스 ID·이름 (띄어쓰기만 다르다). 파일이 있을 때만 본다
    import re

    path = DATA_DIR.parent.parent / "pharmacy-ai" / "cv" / "classes.yaml"
    if not path.exists():
        pytest.skip("pharmacy-ai/cv/classes.yaml이 없다")
    pairs = re.findall(r"id:\s*(\d+),.*?name:\s*([^}\s]+)", path.read_text(encoding="utf-8"))
    cv = {int(i): n for i, n in pairs}
    assert {e.class_id: e.name.replace(" ", "") for e in load_dictionary().entries} == cv


def test_a_path_can_be_given(tmp_path):
    path = tmp_path / "dict.json"
    path.write_text(json.dumps({"drugs": [item()]}, ensure_ascii=False), encoding="utf-8")
    assert [e.item_code for e in load_dictionary(path).entries] == ["900000011"]
    assert [e.item_code for e in load_dictionary(str(path)).entries] == ["900000011"]


def test_a_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="사전 파일이 없습니다"):
        load_dictionary(tmp_path / "없음.json")


@pytest.mark.parametrize("content", [b"{not json", "{}".encode("utf-16")])
def test_a_broken_file_is_a_config_error(tmp_path, content):
    path = tmp_path / "dict.json"
    path.write_bytes(content)
    with pytest.raises(ConfigError, match="JSON\\(UTF-8\\)"):
        load_dictionary(path)


# ------------------------------------------------------------ 찾기


def test_entries_keep_the_file_order_and_a_comparison_key():
    dictionary = dictionary_from_data({"drugs": [item("900000021", " 게보린정 "), item()]})
    assert [e.name for e in dictionary.entries] == ["게보린정", "타이레놀정 500mg"]   # 앞뒤 공백은 지운다
    assert [e.key for e in dictionary.entries] == ["게보린정", "타이레놀정500mg"]


def test_by_item_code():
    dictionary = dictionary_from_data({"drugs": [item("202106092") | {"rx_code": "900000011"}]})
    assert dictionary.by_item_code("202106092").name == "타이레놀정 500mg"
    assert dictionary.by_item_code("900000011") is None          # 처방전 코드로는 찾지 않는다
    assert dictionary.by_item_code(None) is None
    assert dictionary.by_item_code(202106092) is None             # 숫자는 코드 글자가 아니다


def test_by_printed_code_finds_the_rx_code_or_the_item_code():
    dictionary = dictionary_from_data({"drugs": [item("202106092") | {"rx_code": "900000011"}, item("197900277", "게보린정")]})
    assert dictionary.by_printed_code("900000011").item_code == "202106092"     # 처방전 코드
    assert dictionary.by_printed_code("202106092").item_code == "202106092"     # 품목기준코드가 인쇄된 경우
    assert dictionary.by_printed_code("197900277").name == "게보린정"           # rx_code가 없는 약도 품목기준코드로
    assert dictionary.by_printed_code("900000031") is None
    assert dictionary.by_printed_code(None) is None
    assert dictionary.by_printed_code(900000011) is None
    assert dictionary.by_printed_code(["900000011"]) is None     # 글자가 아닌 값이 와도 멈추지 않는다


def test_optional_fields_default_to_none():
    entry = dictionary_from_data({"drugs": [item()]}).entries[0]
    assert (entry.rx_code, entry.class_id) == (None, None)
    entry = dictionary_from_data({"drugs": [item() | {"rx_code": None, "class_id": None}]}).entries[0]
    assert (entry.rx_code, entry.class_id) == (None, None)


def test_entries_cannot_be_changed_after_loading():
    import dataclasses

    entry = dictionary_from_data({"drugs": [item()]}).entries[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.item_code = "900000099"


# ------------------------------------------------------------ 잘못된 사전


@pytest.mark.parametrize("data", [None, [], {}, {"drugs": []}, {"drugs": "타이레놀정"}, {"drugs": {}}])
def test_a_dictionary_without_drugs_is_refused(data):
    with pytest.raises(ConfigError, match="drugs 목록"):
        dictionary_from_data(data)


@pytest.mark.parametrize(
    "bad, said",
    [
        ("타이레놀정", "객체가 아닙니다"),
        (item(code="90000001"), "숫자 9자리"),
        (item(code="9000000110"), "숫자 9자리"),
        (item(code=900000011), "숫자 9자리"),
        (item(code="９００００００１１"), "숫자 9자리"),     # 전각 숫자
        ({"name": "게보린정"}, "숫자 9자리"),
        (item(name=""), "이름이 없습니다"),
        (item(name="   "), "이름이 없습니다"),
        (item(name=None), "이름이 없습니다"),
        (item(name="/ - /"), "비교할 글자"),
        (item() | {"rx_code": "90000001"}, "rx_code는 숫자 9자리"),
        (item() | {"rx_code": 900000011}, "rx_code는 숫자 9자리"),
        (item() | {"rx_code": ""}, "rx_code는 숫자 9자리"),
        (item() | {"class_id": -1}, "class_id는 0 이상의 정수"),
        (item() | {"class_id": "0"}, "class_id는 0 이상의 정수"),
        (item() | {"class_id": 1.0}, "class_id는 0 이상의 정수"),
        (item() | {"class_id": True}, "class_id는 0 이상의 정수"),
    ],
)
def test_a_bad_entry_is_refused_and_numbered(bad, said):
    with pytest.raises(ConfigError, match=said) as error:
        dictionary_from_data({"drugs": [item("900000099", "가나다정"), bad]})
    assert "2번째" in str(error.value)


def test_duplicate_codes_are_refused():
    with pytest.raises(ConfigError, match="2번째.*item_code가 앞 항목의 코드와 겹칩니다"):
        dictionary_from_data({"drugs": [item(), item(name="게보린정")]})


@pytest.mark.parametrize(
    "first, second, said",
    [
        # 어느 코드든 약 하나만 가리켜야 한다: 처방전 코드로 찾을 때 두 약이 걸리면 안 된다
        ({"rx_code": "900000011"}, {"item_code": "100000002", "rx_code": "900000011"}, "rx_code가 다른 코드와 겹칩니다"),
        ({"rx_code": "900000011"}, {"item_code": "900000011"}, "item_code가 앞 항목의 코드와 겹칩니다"),
        ({}, {"item_code": "100000002", "rx_code": "100000001"}, "rx_code가 다른 코드와 겹칩니다"),
        ({"class_id": 0}, {"item_code": "100000002", "class_id": 0}, "class_id가 앞 항목과 겹칩니다"),
    ],
)
def test_codes_must_point_to_one_drug_only(first, second, said):
    a = item("100000001", "가나다정") | first
    b = item("100000002", "라마바정") | second
    with pytest.raises(ConfigError, match="2번째.*" + said):
        dictionary_from_data({"drugs": [a, b]})


def test_an_rx_code_equal_to_its_own_item_code_is_refused():
    with pytest.raises(ConfigError, match="rx_code가 다른 코드와 겹칩니다"):
        dictionary_from_data({"drugs": [item("100000001", "가나다정") | {"rx_code": "100000001"}]})


def test_different_class_ids_and_codes_are_fine():
    d = dictionary_from_data({"drugs": [
        item("100000001", "가나다정") | {"rx_code": "900000001", "class_id": 0},
        item("100000002", "라마바정") | {"rx_code": "900000002", "class_id": 1},
        item("100000003", "사아자정"),
    ]})
    assert [(e.item_code, e.rx_code, e.class_id) for e in d.entries] == [
        ("100000001", "900000001", 0), ("100000002", "900000002", 1), ("100000003", None, None),
    ]


def test_names_that_compare_equal_are_refused():
    # '타이레놀정/500mg'과 '타이레놀정 500mg'은 비교하면 같은 이름이라 어느 약인지 가를 수 없다
    with pytest.raises(ConfigError, match="2번째.*이름이 앞 항목과 같습니다"):
        dictionary_from_data({"drugs": [item(), item("900000012", "타이레놀정/500mg")]})
