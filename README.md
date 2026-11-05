# my_agent_project — 약사 조제 보조 Agent (Agentic AI 파트)

처방전 OCR 텍스트 → 조제지시서 → 약품 매칭 → (약사 승인) → 조제 계획 흐름을 만드는 프로젝트입니다.
지금은 **① 처방전 변환·약품 매칭** 단계까지 들어 있습니다.

## 시작하기

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 열어서 OPENAI_API_KEY를 채우세요
python -m pytest                 # 키 없이도 통과해야 합니다
```

`.python-version`의 3.13으로 테스트했습니다. 코드는 3.10 이상 문법만 씁니다. 다른 버전이면 이 파일을 고쳐 쓰세요.

## 구조 한눈에 보기

| 위치 | 역할 |
|---|---|
| `app/schemas.py` | 조제지시서·약품 줄·LLM 추출 결과 스키마 |
| `app/domain/` | LangGraph·LLM 없이 테스트되는 순수 로직 (이름 정규화, 카탈로그, 매칭, 구조화 변환) |
| `app/nodes/` | domain을 부르는 얇은 노드 (`state` 딕셔너리 받아 바뀐 부분만 돌려줌) |
| `app/llm.py` | gpt-4o 호출 (타임아웃·재시도) |
| `app/prompts/` | 프롬프트 (코드 밖에서 수정) |
| `data/` | 가상 카탈로그·샘플 처방전·정답 파일 |
| `notebooks/test_prompt.ipynb` | 프롬프트 실험 |

## 약품 매칭 원칙

확실할 때만 확정합니다. 정규화한 약품명이 카탈로그 품목 1개와 같고, 처방전 코드가 가리키는 품목과도 모순이 없을 때만
`item_code`(품목기준코드)를 채웁니다. 이름이 비슷한 정도(fuzzy)는 후보로만 보여주고 자동 확정하지 않습니다.
처방전에 인쇄된 9자리 코드는 보험코드일 가능성이 높아서 품목기준코드로 바로 쓰지 않습니다.

## 주의

- `data/drug_catalog.json`은 **가상 코드**입니다. 대상 약품이 확정되면 실제 값으로 바꾸세요.
- 실제 처방전 이미지·OCR 원문은 `data/private/`에만 두세요 (git에서 제외됩니다).
- 노트북은 출력을 지운 뒤 커밋하세요.
