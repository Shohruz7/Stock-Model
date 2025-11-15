"""
Configuration management for Stock Trend Predictor.

Loads settings from config.yaml and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables.

    Environment variables override YAML values.
    Format: SECTION_KEY (e.g., AWS_S3_BUCKET, MODEL_DEFAULT_ESTIMATORS)

    Args:
        config_path: Path to config.yaml (default: project root)

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        # Find config.yaml in project root
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"

    config = {}

    # Load from YAML if exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Override with environment variables
    # Format: SECTION_KEY (e.g., AWS_S3_BUCKET -> aws.s3_bucket)
    for key, value in os.environ.items():
        if "_" in key:
            parts = key.lower().split("_", 1)
            if len(parts) == 2:
                section, subkey = parts
                if section in config and isinstance(config[section], dict):
                    config[section][subkey] = value
                elif section in config:
                    # If section is not a dict, create it
                    config[section] = {subkey: value}

    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "aws.s3_bucket")
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


# Global config instance
_config: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """Get global configuration (lazy-loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Dict[str, Any]:
    """Reload configuration from file."""
    global _config
    _config = load_config()
    return _config




