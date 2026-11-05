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


def test_the_project_dictionary_loads():
    dictionary = load_dictionary()
    assert [(e.item_code, e.name) for e in dictionary.entries] == [
        ("900000011", "타이레놀정 500mg"),
        ("900000021", "게보린정"),
        ("900000041", "트리스펜연질캡슐"),
        ("900000051", "프론크논캡슐"),
    ]
    assert len(dictionary) == 4


def test_dictionary_codes_agree_with_the_ground_truth():
    # 정답 파일에 같은 코드로 나오는 약은 사전과 같은 약이어야 한다 ('/'와 띄어쓰기만 다르다)
    prescriptions = [rx for key, rx in GROUND_TRUTH.items() if key.endswith(".png")]
    assert len(prescriptions) == 5
    truth = {d["code"]: d["name"] for rx in prescriptions for d in rx["drugs"]}
    for entry in load_dictionary().entries:
        assert truth[entry.item_code].replace("/", " ") == entry.name


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


def test_by_code():
    dictionary = dictionary_from_data({"drugs": [item()]})
    assert dictionary.by_code("900000011").name == "타이레놀정 500mg"
    assert dictionary.by_code("900000031") is None
    assert dictionary.by_code(None) is None
    assert dictionary.by_code(900000011) is None        # 숫자는 코드 글자가 아니다
    assert dictionary.by_code(["900000011"]) is None     # 글자가 아닌 값이 와도 멈추지 않는다


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
    ],
)
def test_a_bad_entry_is_refused_and_numbered(bad, said):
    with pytest.raises(ConfigError, match=said) as error:
        dictionary_from_data({"drugs": [item("900000099", "가나다정"), bad]})
    assert "2번째" in str(error.value)


def test_duplicate_codes_are_refused():
    with pytest.raises(ConfigError, match="2번째.*item_code가 앞 항목과 겹칩니다"):
        dictionary_from_data({"drugs": [item(), item(name="게보린정")]})


def test_names_that_compare_equal_are_refused():
    # '타이레놀정/500mg'과 '타이레놀정 500mg'은 비교하면 같은 이름이라 어느 약인지 가를 수 없다
    with pytest.raises(ConfigError, match="2번째.*이름이 앞 항목과 같습니다"):
        dictionary_from_data({"drugs": [item(), item("900000012", "타이레놀정/500mg")]})
