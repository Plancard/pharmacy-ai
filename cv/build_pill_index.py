"""AI Hub 경구약제 라벨 zip → 약품 색인 CSV"""
import zipfile, json, csv, glob
from collections import defaultdict
from pathlib import Path

SRC = Path("data/aihub/166.약품식별_인공지능_개발을_위한_경구약제_이미지_데이터"
           "/01.데이터/1.Training/라벨링데이터/단일경구약제_5000종")
OUT = Path("data/aihub/pill_index.csv")
FIELDS = ["dl_mapping_code", "dl_name", "item_seq", "di_etc_otc_code", "dl_material",
          "drug_shape", "color_class1", "color_class2", "print_front", "print_back",
          "dl_company"]

rows = []
for zp in sorted(SRC.glob("*.zip"), key=lambda p: int(p.name.split("_")[1])):
    with zipfile.ZipFile(zp) as z:
        by_folder = defaultdict(list)
        for n in z.namelist():
            if n.endswith(".json"):
                by_folder[n.split("/")[0]].append(n)
        for folder, files in by_folder.items():
            info = json.load(z.open(files[0]))["images"][0]
            rows.append([zp.name, folder, len(files)] + [info.get(f, "") for f in FIELDS])
    print(f"{zp.name}: {len(by_folder)}종")

rows.sort(key=lambda r: -r[2])  # 이미지 많은 순
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["zip", "folder", "n_images"] + FIELDS)
    w.writerows(rows)
print(f"\n총 {len(rows)}종 → {OUT}")