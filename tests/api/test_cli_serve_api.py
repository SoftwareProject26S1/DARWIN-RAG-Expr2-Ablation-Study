from types import SimpleNamespace
import os
import sys

from darwin_rag_exp2 import cli


def _clear_logging_env(monkeypatch):
    for name in (
        "DARWIN_EXP2_API_LOG_FILE",
        "DARWIN_EXP2_API_LOG_LEVEL",
        "DARWIN_EXP2_API_CAPTURE_STDIO",
        "VLLM_CONFIGURE_LOGGING",
        "VLLM_LOGGING_CONFIG_PATH",
        "VLLM_LOGGING_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_serve_api_cli_invokes_uvicorn(tmp_path, monkeypatch):
    calls = []

    def fake_run(app_target, *, host, port, log_config):
        calls.append(
            {
                "app_target": app_target,
                "host": host,
                "port": port,
                "log_config": log_config,
            }
        )

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    _clear_logging_env(monkeypatch)
    log_file = tmp_path / "serve-api.log"

    result = cli.main(
        [
            "serve-api",
            "--host",
            "0.0.0.0",
            "--port",
            "5070",
            "--log-file",
            str(log_file),
        ]
    )

    assert result == 0
    assert calls == [
        {
            "app_target": "darwin_rag_exp2.api.app:app",
            "host": "0.0.0.0",
            "port": 5070,
            "log_config": None,
        }
    ]
    assert "starting serve-api" in log_file.read_text(encoding="utf-8")
    vllm_log_config = tmp_path / "serve-api.vllm-logging.json"
    assert os.environ["VLLM_LOGGING_CONFIG_PATH"] == str(vllm_log_config)
    assert vllm_log_config.exists()


def test_serve_api_cli_sets_platform_env_for_runtime(tmp_path, monkeypatch):
    calls = []

    def fake_run(app_target, *, host, port, log_config):
        calls.append(
            {
                "app_target": app_target,
                "host": host,
                "port": port,
                "log_config": log_config,
            }
        )

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    _clear_logging_env(monkeypatch)
    monkeypatch.delenv("DARWIN_EXP2_LLM_PLATFORM", raising=False)
    log_file = tmp_path / "serve-api.log"

    result = cli.main(
        [
            "serve-api",
            "--host",
            "0.0.0.0",
            "--port",
            "5070",
            "--platform",
            "ROCm",
            "--log-file",
            str(log_file),
        ]
    )

    assert result == 0
    assert calls == [
        {
            "app_target": "darwin_rag_exp2.api.app:app",
            "host": "0.0.0.0",
            "port": 5070,
            "log_config": None,
        }
    ]
    assert __import__("os").environ["DARWIN_EXP2_LLM_PLATFORM"] == "rocm"
    assert "platform=rocm" in log_file.read_text(encoding="utf-8")
