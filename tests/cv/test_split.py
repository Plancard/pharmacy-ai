"""cv/split.py 그룹 배정 고정 · cv/split_pose.py pose_ 배정 · cv/append_csv.py 덧붙이기."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cv" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


split = load("split")
split_pose = load("split_pose")
append_csv = load("append_csv")


def write_lists(out, groups):
    for s in ("train", "val", "test"):
        lines = [f"/x/images/{g}_s0{k}.jpg" for g, sp in groups.items() if sp == s for k in (1, 2)]
        if s == "val":
            lines.append("/x/images/empty_s01.jpg")
        (out / f"tray_{s}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_groups_from_old_lists(tmp_path):
    write_lists(tmp_path, {"g01": "train", "g02": "val", "g03": "test"})
    assert split.groups_from_lists(tmp_path) == {"g01": "train", "g02": "val", "g03": "test"}


def test_group_in_two_lists_stops(tmp_path):
    write_lists(tmp_path, {"g01": "train", "g02": "val", "g03": "test"})
    (tmp_path / "tray_val.txt").write_text("/x/images/g01_s03.jpg\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        split.groups_from_lists(tmp_path)


def test_fixed_assign_keeps_old_and_sends_new_to_train(tmp_path):
    write_lists(tmp_path, {"g01": "train", "g02": "val", "g03": "test"})
    info = {g: {} for g in ("g01", "g02", "g03", "g21")}
    assign, src, added = split.fixed_assign(tmp_path, info, "train")
    assert assign == {"g01": "train", "g02": "val", "g03": "test", "g21": "train"}
    assert added == ["g21"] and "tray_" in src
    # 다음 실행은 tray_groups.csv를 따른다 (목록이 바뀌어도 배정은 그대로)
    write_lists(tmp_path, {"g01": "test", "g02": "train", "g03": "val"})
    assign2, src2, added2 = split.fixed_assign(tmp_path, info, "train")
    assert assign2 == assign and src2 == "tray_groups.csv" and added2 == []


def test_no_fixed_assign_without_lists(tmp_path):
    assert split.fixed_assign(tmp_path, {"g01": {}}, "train") is None


@pytest.mark.parametrize("name,expected", [
    ("pose_s05", "val"), ("pose_s21", "val"),     # 같은 조건의 짝은 같은 곳
    ("pose_s01", "train"), ("pose_s17", "train"), ("pose_s32", "train"),
])
def test_pose_split_pairs(name, expected):
    assert split_pose.pose_split(name, {5, 7, 8, 12}) == expected


def test_append_merge_fills_and_rejects_duplicates():
    base_cols = ["group", "day", "time", "planned", "actual", "error", "flags"]
    base = [{"group": "g01", "day": "1", "time": "morning", "planned": "0:1", "actual": "0:1", "error": "none", "flags": ""}]
    add_cols = ["group", "day", "time", "actual"]
    rows = append_csv.merge(base_cols, base, add_cols, [{"group": "g21", "day": "1", "time": "noon", "actual": "2:1"}],
                            ["group", "day", "time"], {"error": "none"}, {"planned": "actual"})
    assert rows == [{"group": "g21", "day": "1", "time": "noon", "planned": "2:1", "actual": "2:1",
                     "error": "none", "flags": ""}]
    with pytest.raises(SystemExit):
        append_csv.merge(base_cols, base, add_cols, [{"group": "g01", "day": "1", "time": "morning", "actual": "0:1"}],
                         ["group", "day", "time"], {}, {})
    with pytest.raises(SystemExit):     # 기존 파일에 없는 열
        append_csv.merge(base_cols, base, add_cols + ["oops"], [], ["group"], {}, {})
