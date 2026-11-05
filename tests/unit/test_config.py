from pathlib import Path

import pytest

from hybrid_search.config import Settings, load_dotenv


def test_dotenv_values_do_not_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nEMBEDDING_MODEL="from-file"\nLOG_LEVEL=debug\n\nINVALID LINE\n')
    # set first so that monkeypatch restores the original state after load_dotenv writes it
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.delenv("EMBEDDING_MODEL")
    monkeypatch.setenv("LOG_LEVEL", "warning")

    load_dotenv(env_file)
    settings = Settings.from_env()

    assert settings.embedding_model == "from-file"
    assert settings.log_level == "WARNING"


def test_defaults_match_the_local_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_QUERY_PREFIX",
        "EMBEDDING_API_KEY",
        "HNSW_EF_SEARCH",
        "FUSION_KEYWORD_WEIGHT",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.embedding_dimensions == 768
    assert settings.embedding_query_prefix == "search_query: "
    assert settings.embedding_api_key is None
    assert settings.hnsw_ef_search == 40
    assert settings.fusion_keyword_weight == 1.0


def test_invalid_integers_name_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "sixteen")
    with pytest.raises(ValueError, match="EMBEDDING_BATCH_SIZE"):
        Settings.from_env()


def test_invalid_numbers_name_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUSION_KEYWORD_WEIGHT", "high")
    with pytest.raises(ValueError, match="FUSION_KEYWORD_WEIGHT"):
        Settings.from_env()
