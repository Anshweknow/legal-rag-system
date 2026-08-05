import pytest

from app.config import (
    DEFAULT_OPENROUTER_MODEL,
    ConfigurationError,
    get_settings,
    validate_settings,
)


def test_openrouter_model_defaults_to_supported_free_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    settings = get_settings()

    assert settings.openrouter_model == DEFAULT_OPENROUTER_MODEL


def test_validate_settings_requires_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    settings = get_settings()

    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        validate_settings(settings)


def test_validate_settings_accepts_ingestion_without_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    settings = get_settings()

    validate_settings(settings, require_openrouter=False)
