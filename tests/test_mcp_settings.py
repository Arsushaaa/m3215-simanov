from pathlib import Path

import pytest

from app.config import ConfigError, MCPSettings


def test_mcp_defaults_do_not_read_bot_dotenv_or_require_bot_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "BOT_TOKEN=private-bot\nOPENROUTER_API_KEY=private-llm\nPOSTGRES_PORT=bad\n",
        encoding="utf-8",
    )
    settings = MCPSettings.load(environ={})
    assert settings.postgres_port == 5432
    assert settings.schedule_file == "data/schedule.json"
    assert settings.schedule_timezone == "Europe/Moscow"
    assert settings.mcp_timeout_seconds == 15


def test_mcp_child_environment_is_a_server_only_allowlist(tmp_path):
    path = tmp_path / "server.env"
    path.write_text(
        "POSTGRES_PASSWORD=private-db\nPOSTGRES_PORT=55432\n"
        "SCHEDULE_FILE=/private/local schedule.json\nBOT_TOKEN=private-bot\n"
        "OPENROUTER_API_KEY=private-llm\nARBITRARY_VARIABLE=private-extra\n",
        encoding="utf-8",
    )
    settings = MCPSettings.load(path, environ={"MCP_TIMEOUT_SECONDS": "3"})
    env = settings.environment()
    assert env == {
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "55432",
        "POSTGRES_DB": "bot",
        "POSTGRES_USER": "bot",
        "POSTGRES_PASSWORD": "private-db",
        "SCHEDULE_FILE": "/private/local schedule.json",
        "SCHEDULE_TIMEZONE": "Europe/Moscow",
        "MCP_TIMEOUT_SECONDS": "3",
    }
    assert "private-db" not in repr(settings)
    assert "/private/local" not in repr(settings)
    assert not {"BOT_TOKEN", "OPENROUTER_API_KEY", "ARBITRARY_VARIABLE"} & env.keys()


@pytest.mark.parametrize(
    "field,value",
    [
        ("POSTGRES_PORT", "0"),
        ("POSTGRES_PORT", "65536"),
        ("POSTGRES_PORT", "private-invalid"),
        ("MCP_TIMEOUT_SECONDS", "0"),
        ("MCP_TIMEOUT_SECONDS", "61"),
        ("MCP_TIMEOUT_SECONDS", "private-invalid"),
        ("SCHEDULE_FILE", "https://private-key.example/schedule"),
        ("SCHEDULE_FILE", ""),
        ("SCHEDULE_TIMEZONE", "private-invalid"),
        ("SCHEDULE_TIMEZONE", "localtime"),
        ("SCHEDULE_TIMEZONE", "right/UTC"),
    ],
)
def test_mcp_configuration_errors_are_safe(field, value):
    with pytest.raises(ConfigError) as error:
        MCPSettings.load(environ={field: value})
    assert field in str(error.value)
    assert "private-" not in str(error.value)


def test_mcp_schedule_file_remains_a_local_path_with_spaces(tmp_path):
    missing = tmp_path / "local schedule.json"
    settings = MCPSettings.load(environ={"SCHEDULE_FILE": str(missing)})
    assert Path(settings.schedule_file) == missing
    assert not missing.exists()
