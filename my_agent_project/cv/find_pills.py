"""사용법: python3 cv/find_pills.py 타이레놀 게보린 판콜"""
import csv, sys

rows = list(csv.DictReader(open("data/aihub/pill_index.csv", encoding="utf-8-sig")))
norm = lambda s: s.replace(" ", "").lower()

for q in sys.argv[1:]:
    hits = [r for r in rows if norm(q) in norm(r["dl_name"])]
    print(f"\n■ {q}: {len(hits)}건")
    for r in hits[:10]:
        print(f"  {r['n_images']:>5}장 | {r['dl_name']} | {r['di_etc_otc_code']} | "
              f"{r['drug_shape']}·{r['color_class1']} | 각인 앞'{r['print_front']}' 뒤'{r['print_back']}' | "
              f"품목코드 {r['item_seq']} | {r['zip']}")