"""verify_ingredients.py(사전의 주성분을 저장해 둔 의약품 API 응답과 견주는 스크립트)의 테스트. 네트워크도 키도 쓰지 않는다."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.config import RESOURCES_DIR
from agent.drug_dictionary import DrugEntry, load_dictionary

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "agent" / "verify_ingredients.py"
_spec = importlib.util.spec_from_file_location("verify_ingredients", SCRIPT)
verify_ingredients = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_ingredients)

DICT = load_dictionary(RESOURCES_DIR / "drug_dictionary.json")
ENTRIES = list(DICT.entries)
YOBISIN_HERBS = (
    "Akebia Stem/Alisma Rhizome/Angelica Gigas Root/Cinnamon Bark/Ephedra Herb/Forsythia Fruit/Gardenia Fruit/"
    "Gentian Root and Rhizome/Licorice/Plantago Seed/Raw Ginger/Rehmannia Root/Rhubarb/Scutellaria Root"
)

# 번호 -> (API 제품명, 주성분 칸). 11/11 실제 응답에서 옮겨 적은 값이다.
API_TEXT = {
    0: ("타이레놀정500밀리그람(아세트아미노펜)", "Acetaminophen"),
    1: ("게보린정(수출명:돌로린정)", "Acetaminophen/Anhydrous Caffeine/Isopropylantipyrine"),
    2: ("트리스펜연질캡슐(이부프로펜)", "Ibuprofen"),
    3: ("프론크논캡슐(배농산급탕)", "Baenongsangeubtang Soft Extract (4→1)"),
    4: ("아르펜정", "Arginine/Ibuprofen"),
    5: ("아이월드요비신정", YOBISIN_HERBS),
    6: ("트립라인정", "Bromelain/Crystallized Trypsin"),
    7: ("뮤코다인캡슐200mg(아세틸시스테인)", "Acetylcysteine/Acetylcysteine"),
}


def item(number, **extra):
    name, ingredients = API_TEXT[number]
    return {"ITEM_SEQ": ENTRIES[number].item_code, "ITEM_NAME": name, "ENTP_NAME": "가상제약", "CANCEL_NAME": "정상",
            "ITEM_INGR_NAME": ingredients, **extra}


def saved(number, *items):
    entry = ENTRIES[number]
    return {
        "dictionary": {"item_code": entry.item_code, "name": entry.name},
        "search_term": entry.name,
        "response": {"header": {"resultCode": "00"}, "body": {"items": list(items)}},
    }


def write(folder, number, data):
    path = folder / f"{number:02d}_{ENTRIES[number].item_code}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def folder(tmp_path):
    """8종 모두 저장된 응답이 있는 폴더."""
    for number in range(8):
        write(tmp_path, number, saved(number, item(number)))
    return tmp_path


def run(folder, *extra):
    lines = []
    code = verify_ingredients.main(["--dir", str(folder), *extra], out=lines.append)
    return code, "\n".join(lines)


# ------------------------------------------------------------ 판정


def block(text, number):
    return text.split(f"[{number}] ")[1].split(f"[{number + 1}] ")[0]


def test_the_project_dictionary_matches_the_real_api_response_of_all_eight(folder):
    code, text = run(folder)
    assert code == 0 and text.endswith("전체 8종 중 8종 일치")
    assert text.count("판정: 일치") == 8
    assert "[1] 게보린정 (197900277)\n      사전 주성분: 아세트아미노펜, 이소프로필안티피린, 카페인무수물" in text
    assert "API ITEM_INGR_NAME: Acetaminophen/Anhydrous Caffeine/Isopropylantipyrine" in text


def test_every_ingredient_of_the_dictionary_has_an_english_name():
    for entry in ENTRIES:
        for name in entry.ingredients:
            assert name in verify_ingredients.ENGLISH_NAMES, name


def test_an_ingredient_only_the_api_has_is_named(folder):
    write(folder, 0, saved(0, item(0, ITEM_INGR_NAME="Acetaminophen/Caffeine")))
    code, text = run(folder)
    assert code == 1 and "전체 8종 중 7종 일치" in text
    assert "판정: 일부 일치 → API에만 있는 성분: Caffeine" in block(text, 0)


def test_an_ingredient_only_the_dictionary_has_is_named(folder):
    write(folder, 1, saved(1, item(1, ITEM_INGR_NAME="Acetaminophen")))
    _, text = run(folder)
    assert "판정: 일부 일치 → 사전에만 있는 성분: 이소프로필안티피린, 카페인무수물" in block(text, 1)


def test_nothing_in_common_is_a_mismatch_and_both_sides_are_shown(folder):
    write(folder, 2, saved(2, item(2, ITEM_INGR_NAME="Naproxen")))
    _, text = run(folder)
    assert "판정: 불일치 → 사전에만 있는 성분: 이부프로펜 | API에만 있는 성분: Naproxen" in block(text, 2)


def test_an_ingredient_without_an_english_name_is_pointed_out(folder, monkeypatch):
    monkeypatch.delitem(verify_ingredients._ENGLISH_BY_KEY, "이부프로펜")
    _, text = run(folder)
    assert "사전에만 있는 성분: 이부프로펜 (영어 이름 모름) | API에만 있는 성분: Ibuprofen" in block(text, 2)


def test_an_empty_dictionary_entry_shows_the_api_value_and_does_not_count(folder, monkeypatch):
    emptied = [DrugEntry(e.item_code, e.name, e.key, e.class_id, e.k_code, () if i == 5 else e.ingredients) for i, e in enumerate(ENTRIES)]
    monkeypatch.setattr(verify_ingredients, "load_dictionary", lambda: type("D", (), {"entries": emptied, "__len__": lambda self: 8})())
    code, text = run(folder)
    assert code == 1 and "전체 8종 중 7종 일치" in text
    assert "사전 주성분: (비어 있음)" in block(text, 5) and "Scutellaria Root" in block(text, 5)       # 14종이 잘리지 않고 다 보인다
    assert "판정: 사전 주성분 비어 있음 → 위 API 값을 팀이 확인한 뒤 drug_dictionary.json의 ingredients에 적으세요" in text


def test_every_field_that_looks_like_an_ingredient_is_shown(folder):
    write(folder, 0, saved(0, item(0, ITEM_INGR_CNT="1", MATERIAL_NAME="", ITEM_PERMIT_DATE="20200101")))
    _, text = run(folder)
    assert "API ITEM_INGR_NAME: Acetaminophen" in text and "API ITEM_INGR_CNT: 1" in text
    assert "API MATERIAL_NAME" not in text                                 # 빈 칸은 보이지 않는다
    assert "API ITEM_PERMIT_DATE" not in text                              # 성분이 아닌 칸도 보이지 않는다


def test_a_long_value_is_cut(folder):
    write(folder, 0, saved(0, item(0, MATERIAL_NAME="가" * 900)))
    _, text = run(folder)
    shown = next(line for line in text.splitlines() if line.startswith("      API MATERIAL_NAME: 가"))
    assert shown.endswith("…") and len(shown) < 450


def test_the_field_names_of_the_response_are_shown_once(folder):
    _, text = run(folder)
    assert text.count("응답 필드:") == 1
    assert "응답 필드: ITEM_SEQ, ITEM_NAME, ENTP_NAME, CANCEL_NAME, ITEM_INGR_NAME" in text


def test_the_dictionary_is_not_changed():
    before = (RESOURCES_DIR / "drug_dictionary.json").read_bytes()
    run(Path("없는 폴더"))
    assert (RESOURCES_DIR / "drug_dictionary.json").read_bytes() == before


# ------------------------------------------------------------ 읽을 수 없을 때


def test_a_missing_file_says_to_run_show_api_8_first(tmp_path):
    code, text = run(tmp_path)
    assert code == 1 and "전체 8종 중 0종 일치" in text
    assert text.count("판정: 저장된 응답 없음 → 먼저 python scripts/agent/show_api_8.py 를 돌리세요") == 8


def test_a_broken_file_is_reported_and_the_others_still_run(folder):
    write(folder, 0, saved(0))
    (folder / f"01_{ENTRIES[1].item_code}.json").write_text("{not json", encoding="utf-8")
    write(folder, 2, {"dictionary": {}, "response": "문자열"})
    write(folder, 3, ["목록"])
    _, text = run(folder)
    assert text.count("판정: 응답 파일을 읽을 수 없음") == 3
    assert "판정: 응답에 이 제품 없음" in text                                # 0번: 제품이 하나도 없다
    assert "[4] 아르펜정 (200201239)" in text and "전체 8종 중 4종 일치" in text      # 4 · 5 · 6 · 7번만 일치


def test_a_response_with_another_product_only_does_not_count(folder):
    other = {**item(4), "ITEM_SEQ": "200299999"}
    write(folder, 4, saved(4, other))
    _, text = run(folder)
    block = text.split("[4] ")[1].split("[5] ")[0]
    assert "판정: 응답에 이 제품 없음" in block and "API 제품명" not in block


def test_the_right_product_is_picked_among_several(folder):
    other = {**item(4), "ITEM_SEQ": "200299999", "ITEM_INGR_NAME": "Naproxen"}
    write(folder, 4, saved(4, other, item(4)))
    _, text = run(folder)
    block = text.split("[4] ")[1].split("[5] ")[0]
    assert "판정: 일치" in block


def test_it_needs_neither_a_key_nor_the_network(folder, monkeypatch):
    monkeypatch.delenv("DRUG_API_KEY", raising=False)
    code, text = run(folder)
    assert code == 0 and "전체 8종 중 8종 일치" in text


def test_a_broken_dictionary_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.config.DRUG_DICTIONARY_PATH", tmp_path / "missing.json")
    lines = []
    assert verify_ingredients.main(["--dir", str(tmp_path)], out=lines.append) == 1
    assert lines[0].startswith("설정 문제: 약품 사전 파일이 없습니다")


# ------------------------------------------------------------ 부품


def test_ingredient_fields_keeps_only_filled_fields_with_a_hint_in_the_name():
    got = verify_ingredients.ingredient_fields({
        "ITEM_NAME": "x", "MAIN_ITEM_INGR": " a   b ", "material_name": "c", "INGR_NAME": "", "OTHER_INGR": None, "ATC_CODE": "N02",
        "MATERIAL_FLAG": 0,
    })
    assert got == {"MAIN_ITEM_INGR": "a b", "material_name": "c", "MATERIAL_FLAG": "0"}


def entry_of(*ingredients):
    return DrugEntry("100000001", "가정", "가정", ingredients=ingredients)


def test_api_ingredients_splits_trims_and_drops_repeats():
    assert verify_ingredients.api_ingredients({"ITEM_INGR_NAME": " Arginine / Ibuprofen /ibuprofen/ "}) == ["Arginine", "Ibuprofen"]
    assert verify_ingredients.api_ingredients({"ITEM_INGR_NAME": "Acetylcysteine/Acetylcysteine"}) == ["Acetylcysteine"]
    for empty in ({}, {"ITEM_INGR_NAME": ""}, {"ITEM_INGR_NAME": None}, {"ITEM_INGR_NAME": 3}, {"ITEM_INGR_NAME": " "}):
        assert verify_ingredients.api_ingredients(empty) is None


def test_judge_ignores_order_case_spaces_and_the_note_in_brackets():
    v = verify_ingredients
    assert v.judge(entry_of("L-아르기닌", "이부프로펜"), {"ITEM_INGR_NAME": "IBUPROFEN/arginine"}) == (v.MATCH, [], [])
    assert v.judge(entry_of("배농산급탕연조엑스"), {"ITEM_INGR_NAME": "Baenongsangeubtang Soft Extract (4→1)"}) == (v.MATCH, [], [])
    assert v.judge(entry_of("아세트 아미노펜"), {"ITEM_INGR_NAME": "Acetaminophen"}) == (v.MATCH, [], [])


def test_judge_falls_back_to_korean_text_when_the_response_has_no_ingredient_field():
    v = verify_ingredients
    assert v.judge(entry_of("이부프로펜"), {"ITEM_NAME": "트리스펜연질캡슐(이부프로펜)"}) == (v.MATCH, [], [])
    assert v.judge(entry_of("L-아르기닌", "이부프로펜"), {"ITEM_NAME": "가정", "MATERIAL_NAME": "l - 아르기닌 / 이부 프로펜"}) == (v.MATCH, [], [])
    assert v.judge(entry_of("이부프로펜", "L-아르기닌"), {"ITEM_NAME": "가정", "MATERIAL_NAME": "성분명 : 이부프로펜"}) == (v.PARTIAL, ["L-아르기닌"], [])
    assert v.judge(entry_of("이부프로펜"), {"ITEM_NAME": "가정"}) == (v.DIFFERENT, ["이부프로펜"], [])


def test_judge_of_an_entry_without_ingredients():
    v = verify_ingredients
    assert v.judge(entry_of(), {"ITEM_INGR_NAME": "Ibuprofen"}) == (v.EMPTY, [], [])


def test_the_default_folder_is_the_one_show_api_8_writes_to():
    spec = importlib.util.spec_from_file_location("show_api_8", ROOT / "scripts" / "agent" / "show_api_8.py")
    show_api_8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(show_api_8)
    assert verify_ingredients.DEFAULT_DIR == show_api_8.DEFAULT_OUT == ROOT / "var" / "api"


def test_the_screen_output_can_be_written_in_cp949(folder, tmp_path):
    _, text = run(folder)
    _, none = run(tmp_path / "비어 있음")
    for output in (text, none):
        output.encode("cp949")


def test_it_runs_as_a_script_from_another_folder(tmp_path):
    # 파이프로 받는 출력은 윈도우에서 cp949로 나오므로, 읽을 수 있게 UTF-8로 내보내게 한다 (화면 출력 자체는 위 cp949 시험이 본다)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(tmp_path)], cwd=tmp_path, capture_output=True, env=env,
    )
    out = done.stdout.decode("utf-8", "replace")
    assert done.returncode == 1 and "전체 8종 중 0종 일치" in out
