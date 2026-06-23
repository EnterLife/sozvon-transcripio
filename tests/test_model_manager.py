import os

from speech.model_manager import _model_download_env, _unsupported_proxy_env


def test_unsupported_socks4_proxy_is_detected(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    assert _unsupported_proxy_env() == {"ALL_PROXY": "socks4://127.0.0.1:10808"}


def test_supported_socks5_proxy_is_allowed(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:10808")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    assert _unsupported_proxy_env() == {}


def test_unsupported_windows_system_proxy_is_detected(monkeypatch) -> None:
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(
        "urllib.request.getproxies",
        lambda: {"http": "socks4://127.0.0.1:10808", "https": "socks4://127.0.0.1:10808"},
    )

    assert _unsupported_proxy_env() == {
        "system:http": "socks4://127.0.0.1:10808",
        "system:https": "socks4://127.0.0.1:10808",
    }


def test_model_download_env_temporarily_bypasses_unsupported_proxy(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.setattr("urllib.request.getproxies", lambda: {})

    with _model_download_env():
        assert os.environ["NO_PROXY"] == "*"
        assert os.environ["no_proxy"] == "*"

    assert os.environ["NO_PROXY"] == "localhost"
    if os.name == "nt":
        assert os.environ["no_proxy"] == "localhost"
    else:
        assert "no_proxy" not in os.environ


def test_no_proxy_star_already_bypasses_proxy_validation(monkeypatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks4://127.0.0.1:10808")
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setattr("urllib.request.getproxies", lambda: {"http": "socks4://127.0.0.1:10808"})

    assert _unsupported_proxy_env() == {}
