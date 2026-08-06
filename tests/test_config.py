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


def test_load_dotenv_uses_project_root_env_when_cwd_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    project_env = tmp_path / ".env"
    project_env.write_text("OPENROUTER_API_KEY=sk-or-v1-project-root\n")
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / ".env").write_text("OPENROUTER_API_KEY=wrong-key\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("app.config.PROJECT_DOTENV_PATH", project_env)
    monkeypatch.chdir(other_dir)

    settings = get_settings()

    assert settings.openrouter_api_key == "sk-or-v1-project-root"


def test_log_dotenv_status_reports_key_presence_without_secret_material(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    project_env = tmp_path / ".env"
    project_env.write_text("OPENROUTER_API_KEY=sk-or-v1-secret\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("app.config.PROJECT_DOTENV_PATH", project_env)

    from app.config import log_dotenv_status

    result = log_dotenv_status()

    assert result.path == project_env
    assert result.found is True
    assert result.loaded is True
    assert result.openrouter_api_key_present is True
    assert not hasattr(result, "openrouter_api_key_prefix")
