from types import SimpleNamespace
import sys

from darwin_rag_exp2 import cli


def test_serve_api_cli_invokes_uvicorn(monkeypatch):
    calls = []

    def fake_run(app_target, *, host, port):
        calls.append({"app_target": app_target, "host": host, "port": port})

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))

    result = cli.main(["serve-api", "--host", "0.0.0.0", "--port", "5070"])

    assert result == 0
    assert calls == [
        {
            "app_target": "darwin_rag_exp2.api.app:app",
            "host": "0.0.0.0",
            "port": 5070,
        }
    ]


def test_serve_api_cli_sets_platform_env_for_runtime(monkeypatch):
    calls = []

    def fake_run(app_target, *, host, port):
        calls.append({"app_target": app_target, "host": host, "port": port})

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    monkeypatch.delenv("DARWIN_EXP2_LLM_PLATFORM", raising=False)

    result = cli.main(
        [
            "serve-api",
            "--host",
            "0.0.0.0",
            "--port",
            "5070",
            "--platform",
            "ROCm",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "app_target": "darwin_rag_exp2.api.app:app",
            "host": "0.0.0.0",
            "port": 5070,
        }
    ]
    assert __import__("os").environ["DARWIN_EXP2_LLM_PLATFORM"] == "rocm"
