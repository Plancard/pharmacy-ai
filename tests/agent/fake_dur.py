"""시험용 가짜 DUR (13단계). 진짜 DUR API를 부르지 않고 정해 둔 병용금기 · 효능군을 돌려준다.

    FakeDur      dur.DurLookup 모양 (taboo · efficacy). dur.py · risk.py · 노드 · MCP 서버 시험에 넘긴다
    dur_fetch    dur_api.fetch_rows에 넘기는 가짜 fetch. 실제 응답 모양(response.header/body.items.item)을 흉내 낸다
"""
from agent.dur_api import TabooList, TabooPartner
from agent.errors import DrugApiError

TYLENOL, GEVORIN, TRISPEN, ARPEN = "202106092", "197900277", "201110120", "200201239"
KETOROLAC = "199900001"            # 가상의 케토롤락 제품 (사전 밖)
KETOROLAC_NAME = "가상케토롤락정"
TABOO_REASON = "위장관 출혈 등 이상반응 증가"

# 트리스펜 · 아르펜(이부프로펜)의 병용금기 상대: 케토롤락
TABOOS = {
    TRISPEN: [TabooPartner(KETOROLAC, KETOROLAC_NAME, "케토롤락트로메타민", TABOO_REASON)],
    ARPEN: [TabooPartner(KETOROLAC, KETOROLAC_NAME, "케토롤락트로메타민", TABOO_REASON)],
    KETOROLAC: [TabooPartner(TRISPEN, "트리스펜연질캡슐", "이부프로펜", TABOO_REASON)],
}
# 효능군: 해열진통소염제 (타이레놀 · 게보린 · 트리스펜 · 아르펜)
EFFICACY = {
    TYLENOL: ("해열진통소염제",),
    GEVORIN: ("해열진통소염제",),
    TRISPEN: ("해열진통소염제",),
    ARPEN: ("해열진통소염제",),
}


class FakeDur:
    def __init__(self, taboos=None, efficacy=None, *, fail=(), partial=()):
        self.taboos = TABOOS if taboos is None else taboos
        self.efficacy_map = EFFICACY if efficacy is None else efficacy
        self.fail, self.partial = set(fail), set(partial)
        self.asked: list[tuple[str, str]] = []

    def taboo(self, item_code):
        self.asked.append(("taboo", item_code))
        if item_code in self.fail:
            raise DrugApiError("의약품 API에 연결하지 못했습니다 (URLError).")
        return TabooList(tuple(self.taboos.get(item_code, ())), item_code not in self.partial)

    def efficacy(self, item_code):
        self.asked.append(("efficacy", item_code))
        if item_code in self.fail:
            raise DrugApiError("의약품 API에 연결하지 못했습니다 (URLError).")
        return tuple(self.efficacy_map.get(item_code, ()))


def response(rows, total=None):
    """공공데이터포털 JSON 응답 모양."""
    return {"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                         "body": {"pageNo": 1, "totalCount": len(rows) if total is None else total,
                                  "items": {"item": rows} if rows else ""}}}


def taboo_row(partner_code, partner_name, partner_ingredient, reason=TABOO_REASON, **extra):
    return {"TYPE_NAME": "병용금기", "ITEM_SEQ": "201110120", "ITEM_NAME": "트리스펜연질캡슐", "INGR_KOR_NAME": "이부프로펜",
            "MIXTURE_ITEM_SEQ": partner_code, "MIXTURE_ITEM_NAME": partner_name,
            "MIXTURE_INGR_KOR_NAME": partner_ingredient, "PROHBT_CONTENT": reason, **extra}


def efficacy_row(group, series="", **extra):
    return {"TYPE_NAME": "효능군중복", "ITEM_SEQ": "201110120", "EFFECT_NAME": group, "SERS_NAME": series, **extra}
