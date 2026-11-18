"""이미 찍은 트레이 그룹의 배치(cells.csv planned)대로 가상 처방전을 만든다 (11/18 통합 확인용, 저장소 루트에서 실행).

rx8 처방전 20장과 같은 배치로 찍은 트레이 그룹이 없어서, 거꾸로 '찍어 둔 트레이에 맞는 처방전'을 만든다.
이 처방전을 웹에 올리고 그 그룹 사진을 트레이로 올리면 실제 CV 검수로 '전 칸 일치'·'잘못 담음'을 바로 볼 수 있다.

    python scripts/make_group_rx.py --list               # 처방전으로 만들 수 있는 그룹과 일부러 틀린 칸 수
    python scripts/make_group_rx.py                      # 틀린 칸 없는 그룹 1개 + 틀린 칸 있는 그룹 1개를 골라 만든다
    python scripts/make_group_rx.py --group g07 g09      # 그룹을 직접 고른다

- test 그룹(g03·g06·g08·g10·g11·g20)은 쓰지 않는다 (test 1회 원칙).
- 처방전으로 옮길 수 없는 그룹은 건너뛴다: 같은 약의 1회 개수가 시점마다 다르거나, 날마다 시점이 다르거나, 4일 이상.
- 결과는 data/integration/1118/ (rxg_<그룹>.png). 학습·평가 데이터가 아니라 화면 확인용이다. 저장소에 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "agent"))

from agent.drug_dictionary import load_dictionary  # noqa: E402
from cv.detect import item_codes  # noqa: E402
from cv.tray_inspect import load_truth  # noqa: E402
import make_prescriptions as mp  # noqa: E402

TEST = {"g03", "g06", "g08", "g10", "g11", "g20"}
ORDER = ("morning", "noon", "evening", "bedtime")
OUT = ROOT / "data" / "integration" / "1118"


def drugs_of(cells, cls2name):
    """그룹 planned → [(약 이름, 1회 개수, 시점, 일수)]. 처방전 한 줄로 옮길 수 없으면 None."""
    per = {}                                     # class → {day: {time: count}}
    for (day, time), c in cells.items():
        for cls, n in c["expected"].items():
            if n:
                per.setdefault(cls, {}).setdefault(day, {})[time] = n
    drugs = []
    for cls, days in sorted(per.items()):
        if cls not in cls2name:
            return None
        n_days = max(days)
        if n_days > 3 or sorted(days) != list(range(1, n_days + 1)):
            return None
        slots = {tuple(t for t in ORDER if t in d) for d in days.values()}
        counts = {n for d in days.values() for n in d.values()}
        if len(slots) != 1 or len(counts) != 1:
            return None
        drugs.append((cls2name[cls], counts.pop(), slots.pop(), n_days))
    return drugs or None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cells", default="data/tray/cells.csv")
    p.add_argument("--group", nargs="*")
    p.add_argument("--list", action="store_true")
    p.add_argument("--font", help="한글 글꼴 파일 (기본: Noto CJK, 없으면 /mnt/c/Windows/Fonts/malgun.ttf)")
    a = p.parse_args(argv)

    codes = {e.name: e.item_code for e in load_dictionary().entries}
    code2name = {v: k for k, v in codes.items()}
    cls2name = {cls: code2name[code] for cls, code in item_codes().items() if code in code2name}

    truth = load_truth(a.cells)
    usable = {}
    for g, cells in sorted(truth.items()):
        if g in TEST:
            continue
        drugs = drugs_of(cells, cls2name)
        if drugs:
            usable[g] = (drugs, sum(1 for c in cells.values() if c["error"]))
    if a.list or not usable:
        for g, (drugs, errors) in usable.items():
            print(f"{g}: 틀린 칸 {errors}개 | " + " · ".join(f"{n} {c}알 {'/'.join(t)} {d}일" for n, c, t, d in drugs))
        if not usable:
            print("처방전으로 옮길 수 있는 그룹이 없습니다.")
        return 0

    if a.group:
        picks = [g for g in a.group if g in usable]
        for g in set(a.group) - set(picks):
            print(f"{g}: 쓸 수 없는 그룹 (test 그룹이거나 처방전으로 옮길 수 없음)")
    else:
        clean = sorted((g for g, (_, e) in usable.items() if e == 0), key=lambda g: -len(usable[g][0]))   # 약 가짓수 많은 그룹부터
        wrong = [g for g, (_, e) in usable.items() if e > 0]
        picks = clean[:1] + wrong[:1]

    fonts = mp.DEFAULT_FONTS + [("/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf")]   # WSL에서는 Windows 글꼴
    font, bold = next(((f, b) for f, b in fonts if Path(f).exists()), (None, None))
    if a.font:
        font, bold = a.font, None
    if font is None:
        print("한글 글꼴을 찾지 못했습니다. --font 로 글꼴 파일을 알려 주세요.")
        return 1
    if bold and not Path(bold).exists():
        bold = None
    OUT.mkdir(parents=True, exist_ok=True)
    for i, g in enumerate(picks):
        drugs, errors = usable[g]
        rx = {"id": f"rxg_{g}", "dept": "가정의학과", "patient": "테스트", "disease": "R51", "doctor": "이테스트",
              "hospital": mp.HOSPITAL, "time": f"14:{20 + i:02d}:00", "drugs": drugs}
        path = OUT / f"rxg_{g}.png"
        mp.draw(rx, 90 + i, codes, path, font, bold)
        kind = "전 칸 일치 확인용" if errors == 0 else f"잘못 담음 확인용 (틀린 칸 {errors}개)"
        print(f"{path.relative_to(ROOT)}: {kind} — 트레이는 {g} 사진")
        for n, c, t, d in drugs:
            print(f"    {n} {c}알 · {'/'.join(t)} · {d}일")
    return 0


if __name__ == "__main__":
    sys.exit(main())
