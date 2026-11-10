"""설정 (agent/config.py). 실제 .env와 환경변수에 영향받지 않도록 매번 비우고 다시 읽는다."""
import importlib

import pytest

from agent import config
from agent.errors import ConfigError

NAMES = ("OPENAI_API_KEY", "OPENAI_MODEL", "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES", "DRUG_API_KEY",
         "DRUG_API_TIMEOUT_SECONDS", "DRUG_API_URL", "DRUG_DICTIONARY_PATH", "DATABASE_URL")


@pytest.fixture
def fresh(monkeypatch):
    """.env를 읽지 않고 환경변수를 모두 비운 상태로 config를 다시 불러온다."""
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    yield lambda: importlib.reload(config)
    importlib.reload(config)    # 다른 테스트에 바뀐 값이 남지 않게 원래대로


def test_defaults_without_any_env(fresh):
    c = fresh()
    assert c.OPENAI_MODEL == "gpt-5-mini-2025-08-07"
    assert (c.LLM_TIMEOUT_SECONDS, c.LLM_MAX_RETRIES, c.DRUG_API_TIMEOUT_SECONDS) == (20.0, 1, 5.0)
    assert c.DRUG_API_URL.endswith("/DrugPrdtPrmsnInfoService08/getDrugPrdtPrmsnInq08")


def test_the_dictionary_lives_in_resources_not_data(fresh):
    c = fresh()
    assert c.RESOURCES_DIR.name == "resources" and c.RESOURCES_DIR.parent.name == "agent"
    assert c.DRUG_DICTIONARY_PATH == c.RESOURCES_DIR / "drug_dictionary.json"
    assert "data" not in c.DRUG_DICTIONARY_PATH.parts        # .gitignore의 data/ 규칙에 걸리는 이름


def test_env_overrides(fresh, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    monkeypatch.setenv("DRUG_DICTIONARY_PATH", str(tmp_path / "d.json"))
    c = fresh()
    assert (c.OPENAI_MODEL, c.LLM_MAX_RETRIES, c.DRUG_DICTIONARY_PATH) == ("gpt-x", 3, tmp_path / "d.json")


def test_a_blank_dictionary_path_falls_back_to_the_default(fresh, monkeypatch):
    monkeypatch.setenv("DRUG_DICTIONARY_PATH", "")           # .env.example처럼 빈 값
    c = fresh()
    assert c.DRUG_DICTIONARY_PATH == c.RESOURCES_DIR / "drug_dictionary.json"


def test_keys_are_optional_for_the_drug_api_and_required_when_asked(fresh, monkeypatch):
    c = fresh()
    assert c.drug_api_key() is None
    with pytest.raises(ConfigError, match="DRUG_API_KEY"):
        c.require_drug_api_key()
    monkeypatch.setenv("DRUG_API_KEY", "  abc  ")
    assert c.drug_api_key() == "abc" and c.require_drug_api_key() == "abc"


def test_the_openai_key_is_required_when_asked(fresh, monkeypatch):
    c = fresh()
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        c.require_openai_key()
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert c.require_openai_key() == "k"


def test_database_url(fresh, monkeypatch):
    c = fresh()
    assert c.database_url() is None
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
    assert c.database_url() == "postgresql://u:p@localhost:5432/db"
