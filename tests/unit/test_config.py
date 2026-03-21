"""Tests for configuration loading and validation."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from scopescrape.config import (
    DEFAULTS,
    _apply_env_overrides,
    _deep_merge,
    _resolve_env_vars,
    load_config,
    validate_config,
)


class TestDeepMerge:
    def test_flat_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"scoring": {"weights": {"frequency": 0.25}, "min_score": 5.0}}
        override = {"scoring": {"weights": {"frequency": 0.30}}}
        result = _deep_merge(base, override)
        assert result["scoring"]["weights"]["frequency"] == 0.30
        assert result["scoring"]["min_score"] == 5.0

    def test_base_unmodified(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1  # original unchanged


class TestResolveEnvVars:
    def test_resolve_set_var(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        config = {"key": "${MY_KEY}"}
        result = _resolve_env_vars(config)
        assert result["key"] == "secret123"

    def test_resolve_unset_var(self):
        config = {"key": "${NONEXISTENT_VAR_XYZ}"}
        result = _resolve_env_vars(config)
        assert result["key"] == ""

    def test_non_template_unchanged(self):
        config = {"key": "plain_value"}
        result = _resolve_env_vars(config)
        assert result["key"] == "plain_value"

    def test_nested_resolution(self, monkeypatch):
        monkeypatch.setenv("NESTED_VAL", "deep")
        config = {"outer": {"inner": "${NESTED_VAL}"}}
        result = _resolve_env_vars(config)
        assert result["outer"]["inner"] == "deep"


class TestApplyEnvOverrides:
    def test_reddit_override(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "env_id")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "env_secret")
        config = {"reddit": {"client_id": "", "client_secret": ""}}
        result = _apply_env_overrides(config)
        assert result["reddit"]["client_id"] == "env_id"
        assert result["reddit"]["client_secret"] == "env_secret"

    def test_no_override_when_unset(self):
        config = {"reddit": {"client_id": "yaml_id"}}
        # Clear env vars that might exist
        for key in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
            os.environ.pop(key, None)
        result = _apply_env_overrides(config)
        assert result["reddit"]["client_id"] == "yaml_id"


class TestLoadConfig:
    def test_loads_defaults_when_no_file(self):
        config = load_config("/nonexistent/path/config.yaml")
        assert "reddit" in config
        assert "scoring" in config
        assert config["scoring"]["weights"]["frequency"] == 0.25

    def test_loads_yaml_file(self, tmp_path):
        yaml_content = {
            "scoring": {"min_score": 3.0},
            "storage": {"retention_hours": 24},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(yaml_content))

        config = load_config(str(config_file))
        assert config["scoring"]["min_score"] == 3.0
        assert config["storage"]["retention_hours"] == 24
        # Defaults still present
        assert config["scoring"]["weights"]["frequency"] == 0.25


class TestValidateConfig:
    def test_valid_config(self, sample_config):
        errors = validate_config(sample_config)
        assert errors == []

    def test_missing_reddit_credentials(self):
        config = DEFAULTS.copy()
        errors = validate_config(config, require_reddit=True)
        assert len(errors) == 2
        assert any("client_id" in e for e in errors)
        assert any("client_secret" in e for e in errors)

    def test_skip_reddit_validation(self):
        config = DEFAULTS.copy()
        errors = validate_config(config, require_reddit=False)
        assert errors == []

    def test_bad_weights_sum(self, sample_config):
        sample_config["scoring"]["weights"]["frequency"] = 0.5
        errors = validate_config(sample_config)
        assert any("weights sum" in e for e in errors)

    def test_bad_min_score(self, sample_config):
        sample_config["scoring"]["min_score"] = 15.0
        errors = validate_config(sample_config)
        assert any("min_score" in e for e in errors)

    def test_bad_retention(self, sample_config):
        sample_config["storage"]["retention_hours"] = 0
        errors = validate_config(sample_config)
        assert any("retention_hours" in e for e in errors)
