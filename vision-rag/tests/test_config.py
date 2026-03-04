"""Tests for configuration and settings."""

from proxy.config import ModelBackend, Settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.port == 8000
        assert s.debug is False
        assert s.default_top_k == 5
        assert s.default_dpi == 300

    def test_backends_property(self):
        s = Settings()
        backends = s.backends
        assert "retrieve" in backends
        assert "extract" in backends
        assert "generate" in backends
        assert backends["retrieve"].name == "colpali"
        assert backends["extract"].name == "qwen3vl"
        assert backends["generate"].name == "qwen25"

    def test_backend_urls(self):
        s = Settings()
        assert s.backends["retrieve"].url == s.colpali_url
        assert s.backends["extract"].url == s.qwen3vl_url
        assert s.backends["generate"].url == s.qwen25_url

    def test_cors_origins_default_empty(self):
        s = Settings()
        assert s.cors_origins == []

    def test_circuit_breaker_defaults(self):
        s = Settings()
        assert s.circuit_failure_threshold == 5
        assert s.circuit_recovery_timeout == 30

    def test_override_via_env(self, monkeypatch):
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")
        s = Settings()
        assert s.port == 9000
        assert s.debug is True


class TestModelBackend:
    def test_defaults(self):
        mb = ModelBackend(name="test", url="http://test:8000", model_id="test/model", port=8000)
        assert mb.health_path == "/health"
        assert mb.timeout == 120.0
