"""Configuration loader for ScopeScrape.

Loads settings from YAML config file, falls back to environment variables
and .env file, and applies sensible defaults for anything unset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


# Default config values
DEFAULTS: dict[str, Any] = {
    "reddit": {
        "client_id": "",
        "client_secret": "",
        "user_agent": "ScopeScrape v0.1",
        "rate_limit_delay": 1.0,
        "comment_depth": 3,
        "replace_more_limit": 10,
    },
    "hn": {
        "rate_limit_delay": 0.1,
        "comment_depth": 5,
    },
    "scoring": {
        "weights": {
            "frequency": 0.25,
            "intensity": 0.20,
            "specificity": 0.25,
            "recency": 0.30,
        },
        "min_score": 5.0,
    },
    "storage": {
        "db_path": "~/.scopescrape/data.db",
        "retention_hours": 48,
        "in_memory": False,
    },
    "scan": {
        "default_limit": 100,
        "default_time_range": "week",
        "default_platforms": ["reddit"],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, preferring override values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_env_vars(config: dict) -> dict:
    """Replace ${ENV_VAR} placeholders with actual environment variable values."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            resolved[key] = os.environ.get(env_key, "")
        else:
            resolved[key] = value
    return resolved


def _apply_env_overrides(config: dict) -> dict:
    """Override Reddit credentials from environment variables if set."""
    env_mappings = {
        "REDDIT_CLIENT_ID": ("reddit", "client_id"),
        "REDDIT_CLIENT_SECRET": ("reddit", "client_secret"),
        "REDDIT_USER_AGENT": ("reddit", "user_agent"),
    }

    for env_key, (section, field) in env_mappings.items():
        env_val = os.environ.get(env_key)
        if env_val:
            if section not in config:
                config[section] = {}
            config[section][field] = env_val

    return config


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from YAML file, env vars, and defaults.

    Resolution order (later wins):
      1. Built-in defaults
      2. YAML config file
      3. .env file
      4. Environment variables

    Args:
        path: Path to YAML config file. If None, checks
              ~/.scopescrape/config.yaml, then ./config.yaml.

    Returns:
        Merged configuration dict.
    """
    # Load .env file if present
    load_dotenv()

    # Start with defaults
    config = DEFAULTS.copy()

    # Find and load YAML config
    yaml_path = _find_config_file(path)
    if yaml_path:
        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f) or {}
        yaml_config = _resolve_env_vars(yaml_config)
        config = _deep_merge(config, yaml_config)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def _find_config_file(explicit_path: Optional[str]) -> Optional[Path]:
    """Locate config file by explicit path or default search locations."""
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.exists():
            return p
        return None

    search_paths = [
        Path("config.yaml"),
        Path("config/config.yaml"),
        Path.home() / ".scopescrape" / "config.yaml",
    ]

    for p in search_paths:
        if p.exists():
            return p

    return None


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""


def validate_config(config: dict, require_reddit: bool = True) -> list[str]:
    """Validate configuration and return a list of issues.

    Args:
        config: The merged config dict.
        require_reddit: Whether Reddit credentials are required.

    Returns:
        List of validation error strings. Empty list means valid.
    """
    errors = []

    if require_reddit:
        reddit = config.get("reddit", {})
        if not reddit.get("client_id"):
            errors.append(
                "Reddit client_id is missing. Set REDDIT_CLIENT_ID env var "
                "or add reddit.client_id to config.yaml."
            )
        if not reddit.get("client_secret"):
            errors.append(
                "Reddit client_secret is missing. Set REDDIT_CLIENT_SECRET env var "
                "or add reddit.client_secret to config.yaml."
            )

    # Validate scoring weights sum to ~1.0
    weights = config.get("scoring", {}).get("weights", {})
    if weights:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"Scoring weights sum to {total:.2f}, expected 1.0. "
                f"Current: {weights}"
            )

    # Validate min_score range
    min_score = config.get("scoring", {}).get("min_score", 5.0)
    if not (0.0 <= min_score <= 10.0):
        errors.append(f"min_score must be between 0.0 and 10.0, got {min_score}")

    # Validate retention_hours
    retention = config.get("storage", {}).get("retention_hours", 48)
    if retention < 1:
        errors.append(f"retention_hours must be >= 1, got {retention}")

    return errors
