"""Configuration management utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration container with attribute-style access.

    Args:
        config_dict: Dictionary of configuration parameters
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute access.

        Args:
            name: Configuration parameter name

        Returns:
            Configuration value

        Raises:
            AttributeError: If parameter not found
        """
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value

        raise AttributeError(f"Config has no parameter '{name}'")

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by dictionary access.

        Args:
            key: Configuration parameter name

        Returns:
            Configuration value
        """
        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default.

        Args:
            key: Configuration parameter name
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary of configuration parameters
        """
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration object

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        config_dict = {}

    return Config(config_dict)


def save_config(config: Config, output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving configuration to {output_path}")

    config_dict = config.to_dict()

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("Configuration saved successfully")


def merge_configs(base_config: Config, override_config: Config) -> Config:
    """Merge two configurations with override priority.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    base_dict = base_config.to_dict()
    override_dict = override_config.to_dict()

    merged = _deep_merge(base_dict, override_dict)

    return Config(merged)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
