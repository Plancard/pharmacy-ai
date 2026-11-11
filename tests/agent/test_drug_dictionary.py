"""약품 사전 읽기 (agent/drug_dictionary.py)."""
import dataclasses
import json
import re
from pathlib import Path

import pytest

from agent import config
from agent.config import RESOURCES_DIR
from agent.drug_dictionary import DrugDictionary, DrugEntry, dictionary_from_data, ingredient_key, load_dictionary
from agent.errors import ConfigError

DICTIONARY_FILE = RESOURCES_DIR / "drug_dictionary.json"
CLASSES = Path(__file__).resolve().parents[2] / "cv" / "classes.yaml"

# 대상 약 8종: (클래스 ID, 품목기준코드, 이름). 이름·순서는 CV의 cv/classes.yaml과 같다.
# 아이월드요비신정의 코드(200713170)는 명세서 2.6절 값이고 아직 의약품 API로 확인하지 않았다 (5단계 verify_dictionary).
TARGET_8 = [
    (0, "202106092", "타이레놀정500mg"),
    (1, "197900277", "게보린정"),
    (2, "201110120", "트리스펜연질캡슐"),
    (3, "200200099", "프론크논캡슐"),
    (4, "200201239", "아르펜정"),
    (5, "200713170", "아이월드요비신정 416.7mg"),
    (6, "200002714", "트립라인정"),
    (7, "200308774", "뮤코다인캡슐200mg"),
]


# 클래스 ID -> (K 코드, 주성분). K 코드는 cv/classes.yaml, 주성분은 명세서 2.6절 표에서 시작해
# 11/11 의약품 API 응답과 맞춘 값이다 (트립라인에 결정트립신 추가, 요비신정 한약재 14종). scripts/agent/verify_ingredients.py로 확인한다.
K_CODES_AND_INGREDIENTS = {
    0: ("K-004378", ("아세트아미노펜",)),
    1: ("K-000573", ("아세트아미노펜", "이소프로필안티피린", "카페인무수물")),
    2: ("K-029163", ("이부프로펜",)),
    3: ("K-011425", ("배농산급탕연조엑스",)),
    4: ("K-011783", ("이부프로펜", "L-아르기닌")),
    5: ("K-022749", ("목통", "택사", "당귀", "육계", "마황", "연교", "치자", "용담", "감초", "차전자", "생강", "지황", "대황", "황금")),
    6: ("K-009379", ("브로멜라인", "결정트립신")),
    7: ("K-013969", ("아세틸시스테인",)),
}


def item(code="100000001", name="가나다정", **extra):
    return {"item_code": code, "name": name, **extra}


# ------------------------------------------------------------ 실제 사전 파일


def test_the_project_dictionary_is_the_eight_target_drugs():
    dictionary = load_dictionary(DICTIONARY_FILE)
    assert [(e.class_id, e.item_code, e.name) for e in dictionary.entries] == TARGET_8
    assert len(dictionary) == 8


def test_dictionary_is_the_same_eight_drugs_the_samples_use():
    """샘플(test_schema.TARGET_8)과 사전의 품목기준코드·이름이 같다. 한쪽만 고치면 여기서 알린다."""
    from test_schema import TARGET_8

    assert {e.item_code: e.name for e in load_dictionary(DICTIONARY_FILE).entries} == TARGET_8


def test_the_default_dictionary_file_lives_in_the_resources_folder():
    assert DICTIONARY_FILE.exists()
    assert "data" not in DICTIONARY_FILE.parts            # .gitignore의 data/ 규칙에 걸리는 이름
    assert config.RESOURCES_DIR / "drug_dictionary.json" == DICTIONARY_FILE


def test_load_without_a_path_uses_the_configured_file(monkeypatch, tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"drugs": [item()]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(config, "DRUG_DICTIONARY_PATH", path)
    assert [e.item_code for e in load_dictionary().entries] == ["100000001"]


def test_the_dictionary_file_has_no_prescription_code_column():
    # 처방전 코드(rx_code) 칸은 없다. 자동 확정은 인쇄된 코드가 품목기준코드와 같을 때뿐이다 (drug_match.py)
    data = json.loads(DICTIONARY_FILE.read_text(encoding="utf-8"))
    assert all(set(d) == {"class_id", "item_code", "name", "k_code", "ingredients"} for d in data["drugs"])


def test_the_project_dictionary_has_the_k_code_and_ingredients_of_every_drug():
    dictionary = load_dictionary(DICTIONARY_FILE)
    got = {e.class_id: (e.k_code, e.ingredients) for e in dictionary.entries}
    assert got == K_CODES_AND_INGREDIENTS


def test_every_target_drug_has_its_ingredients():
    assert all(e.ingredients for e in load_dictionary(DICTIONARY_FILE).entries)


def test_only_the_two_demo_pairs_share_an_ingredient():
    """8종 가운데 성분이 겹치는 쌍은 시연 조합 A · B뿐이다. 사전을 고치다 뜻밖의 중복이 생기면 여기서 알린다."""
    entries = load_dictionary(DICTIONARY_FILE).entries
    pairs = [
        (a.class_id, b.class_id)
        for i, a in enumerate(entries) for b in entries[i + 1:]
        if {ingredient_key(x) for x in a.ingredients} & {ingredient_key(x) for x in b.ingredients}
    ]
    assert pairs == [(0, 1), (2, 4)]


def test_the_demo_combinations_share_one_ingredient_each():
    dictionary = load_dictionary(DICTIONARY_FILE)
    by_class = {e.class_id: {ingredient_key(i) for i in e.ingredients} for e in dictionary.entries}
    assert by_class[0] & by_class[1] == {"아세트아미노펜"}        # 조합 A: 타이레놀 · 게보린
    assert by_class[2] & by_class[4] == {"이부프로펜"}            # 조합 B: 트리스펜 · 아르펜


def test_the_note_of_the_file_tells_where_the_ingredients_came_from():
    note = json.loads(DICTIONARY_FILE.read_text(encoding="utf-8"))["note"]
    assert "명세서 2.6절" in note and "verify_ingredients" in note and "ITEM_INGR_NAME" in note


@pytest.mark.skipif(not CLASSES.exists(), reason="cv/classes.yaml 이 없다")
def test_dictionary_equals_the_cv_classes_file():
    """클래스 ID와 이름이 CV의 classes.yaml(확정 알약 8종)과 같다. 어긋나면 사전이나 CV 쪽을 먼저 맞춘다."""
    rows = re.findall(r"\{id: (\d+), k_code: (K-\d+), name: (.+?)\}", CLASSES.read_text(encoding="utf-8"))
    assert [(int(i), name) for i, _, name in rows] == [(e.class_id, e.name) for e in load_dictionary(DICTIONARY_FILE).entries]


@pytest.mark.skipif(not CLASSES.exists(), reason="cv/classes.yaml 이 없다")
def test_the_k_codes_equal_the_cv_classes_file():
    """k_code가 CV의 classes.yaml과 같다. CV가 k_code로 약을 보내면 사전이 품목기준코드를 찾아 준다 (by_k_code)."""
    rows = re.findall(r"\{id: (\d+), k_code: (K-\d+), name: (.+?)\}", CLASSES.read_text(encoding="utf-8"))
    dictionary = load_dictionary(DICTIONARY_FILE)
    assert [(int(i), k) for i, k, _ in rows] == [(e.class_id, e.k_code) for e in dictionary.entries]
    for _, k_code, name in rows:
        assert dictionary.by_k_code(k_code).name == name


# ------------------------------------------------------------ 파일 읽기


def test_a_path_can_be_given(tmp_path):
    path = tmp_path / "dict.json"
    path.write_text(json.dumps({"drugs": [item()]}, ensure_ascii=False), encoding="utf-8")
    assert [e.item_code for e in load_dictionary(path).entries] == ["100000001"]
    assert [e.item_code for e in load_dictionary(str(path)).entries] == ["100000001"]


def test_a_note_next_to_drugs_is_allowed(tmp_path):
    path = tmp_path / "dict.json"
    path.write_text(json.dumps({"note": "메모", "drugs": [item()]}, ensure_ascii=False), encoding="utf-8")
    assert len(load_dictionary(path)) == 1


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
    dictionary = dictionary_from_data({"drugs": [item("197900277", " 게보린정 "), item("202106092", "타이레놀정 500mg")]})
    assert [e.name for e in dictionary.entries] == ["게보린정", "타이레놀정 500mg"]   # 앞뒤 공백은 지운다
    assert [e.key for e in dictionary.entries] == ["게보린정", "타이레놀정500mg"]


def test_by_item_code():
    dictionary = dictionary_from_data({"drugs": [item("202106092", "타이레놀정500mg")]})
    assert dictionary.by_item_code("202106092").name == "타이레놀정500mg"
    assert dictionary.by_item_code("900000011") is None
    assert dictionary.by_item_code(None) is None
    assert dictionary.by_item_code(202106092) is None            # 숫자는 코드 글자가 아니다
    assert dictionary.by_item_code(["202106092"]) is None        # 글자가 아닌 값이 와도 멈추지 않는다


def test_by_class_id():
    dictionary = load_dictionary(DICTIONARY_FILE)
    assert dictionary.by_class_id(1).name == "게보린정"
    assert dictionary.by_class_id(8) is None
    assert dictionary.by_class_id(None) is None
    assert dictionary.by_class_id(True) is None                  # True는 1이 아니다
    assert dictionary.by_class_id("1") is None


def test_by_k_code():
    dictionary = load_dictionary(DICTIONARY_FILE)
    assert dictionary.by_k_code("K-000573").item_code == "197900277"
    assert dictionary.by_k_code("K-999999") is None
    assert dictionary.by_k_code("k-000573") is None              # 대소문자까지 같아야 한다
    assert dictionary.by_k_code(None) is None
    assert dictionary.by_k_code(573) is None
    assert dictionary.by_k_code(["K-000573"]) is None


def test_k_code_and_ingredients_are_optional():
    entry = dictionary_from_data({"drugs": [item()]}).entries[0]
    assert (entry.k_code, entry.ingredients) == (None, ())
    entry = dictionary_from_data({"drugs": [item(k_code=None, ingredients=None)]}).entries[0]
    assert (entry.k_code, entry.ingredients) == (None, ())


def test_ingredients_are_trimmed_and_kept_in_order():
    entry = dictionary_from_data({"drugs": [item(ingredients=[" 아세트아미노펜 ", "카페인무수물"])]}).entries[0]
    assert entry.ingredients == ("아세트아미노펜", "카페인무수물")


@pytest.mark.parametrize("a, b", [
    ("아세트아미노펜", "아세트 아미노펜"),
    ("L-아르기닌", "ｌ-아르기닌"),          # 전각 · 대소문자
    ("Ibuprofen", "ibuprofen"),
])
def test_ingredient_key_ignores_spaces_width_and_case(a, b):
    assert ingredient_key(a) == ingredient_key(b)


def test_ingredient_key_keeps_different_ingredients_apart():
    assert ingredient_key("이부프로펜") != ingredient_key("덱시부프로펜")
    assert ingredient_key("아세트아미노펜") != ingredient_key("파라세타몰")   # 다른 이름은 사전에 한 가지로 적는다


def test_class_id_is_optional_and_defaults_to_none():
    entry = dictionary_from_data({"drugs": [item()]}).entries[0]
    assert entry.class_id is None
    assert dictionary_from_data({"drugs": [item(class_id=None)]}).entries[0].class_id is None


def test_entries_cannot_be_changed_after_loading():
    entry = dictionary_from_data({"drugs": [item()]}).entries[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.item_code = "900000099"


def test_an_empty_dictionary_can_be_built_by_hand():
    assert len(DrugDictionary([])) == 0
    assert DrugDictionary([]).by_item_code("202106092") is None
    assert DrugDictionary([DrugEntry("100000001", "가나다정", "가나다정")]).by_item_code("100000001").name == "가나다정"


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
        (item(code="100000002\n"), "숫자 9자리"),            # 줄바꿈이 붙은 코드
        ({"name": "게보린정"}, "숫자 9자리"),
        (item("100000002", name=""), "이름이 없습니다"),
        (item("100000002", name="   "), "이름이 없습니다"),
        (item("100000002", name=None), "이름이 없습니다"),
        (item("100000002", name="/ - /"), "비교할 글자"),
        (item("100000002", class_id=-1), "class_id는 0 이상의 정수"),
        (item("100000002", class_id="0"), "class_id는 0 이상의 정수"),
        (item("100000002", class_id=1.0), "class_id는 0 이상의 정수"),
        (item("100000002", class_id=True), "class_id는 0 이상의 정수"),
        (item("100000002", k_code="K-12345"), "k_code는 K-와 숫자 6자리"),
        (item("100000002", k_code="k-000573"), "k_code는 K-와 숫자 6자리"),
        (item("100000002", k_code="K-０００５７３"), "k_code는 K-와 숫자 6자리"),     # 전각 숫자
        (item("100000002", k_code="K-000573\n"), "k_code는 K-와 숫자 6자리"),
        (item("100000002", k_code=573), "k_code는 K-와 숫자 6자리"),
        (item("100000002", ingredients="아세트아미노펜"), "ingredients는 성분 이름"),
        (item("100000002", ingredients=[1]), "ingredients는 성분 이름"),
        (item("100000002", ingredients=[""]), "ingredients는 성분 이름"),
        (item("100000002", ingredients=["  "]), "ingredients는 성분 이름"),
        (item("100000002", ingredients=["이부프로펜", "이부 프로펜"]), "같은 성분이 두 번"),
        (item("100000002", rx_code="900000011"), "알 수 없는 칸이 있습니다: rx_code"),     # 예전 사전의 칸이 남아 있으면 알린다
        (item("100000002", memo="x", rx_code="9"), "알 수 없는 칸이 있습니다: memo, rx_code"),
    ],
)
def test_a_bad_entry_is_refused_and_numbered(bad, said):
    with pytest.raises(ConfigError, match=said) as error:
        dictionary_from_data({"drugs": [item("100000001", "가나다정"), bad]})
    assert "2번째" in str(error.value)


def test_duplicate_codes_are_refused():
    with pytest.raises(ConfigError, match="2번째.*item_code가 앞 항목의 코드와 겹칩니다"):
        dictionary_from_data({"drugs": [item(), item(name="게보린정")]})


def test_names_that_compare_equal_are_refused():
    # '타이레놀정/500mg'과 '타이레놀정 500mg'은 비교하면 같은 이름이라 어느 약인지 가를 수 없다
    with pytest.raises(ConfigError, match="2번째.*이름이 앞 항목과 같습니다"):
        dictionary_from_data({"drugs": [item("100000001", "타이레놀정 500mg"), item("100000002", "타이레놀정/500mg")]})


def test_a_class_id_used_twice_is_refused():
    with pytest.raises(ConfigError, match="2번째.*class_id가 앞 항목과 겹칩니다"):
        dictionary_from_data({"drugs": [item(class_id=0), item("100000002", "라마바정", class_id=0)]})


def test_a_k_code_used_twice_is_refused():
    with pytest.raises(ConfigError, match="2번째.*k_code가 앞 항목과 겹칩니다"):
        dictionary_from_data({"drugs": [item(k_code="K-000573"), item("100000002", "라마바정", k_code="K-000573")]})


def test_different_class_ids_and_codes_are_fine():
    d = dictionary_from_data({"drugs": [
        item("100000001", "가나다정", class_id=0),
        item("100000002", "라마바정", class_id=1),
        item("100000003", "사아자정"),
        item("100000004", "차카타정"),
    ]})
    assert [(e.item_code, e.class_id) for e in d.entries] == [
        ("100000001", 0), ("100000002", 1), ("100000003", None), ("100000004", None),
    ]
