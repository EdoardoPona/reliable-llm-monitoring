"""Configuration utilities for experiments."""

import os
import re
from types import SimpleNamespace

import yaml


def expand_env_vars(value):
    """Expand environment variables in a value (string or recursively in dict/list)."""
    if isinstance(value, str):
        # Replace ${VAR_NAME} with environment variable values
        def replace_env_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace_env_var, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    else:
        return value


def load_config(config_path):
    """Load configuration from a YAML file and expand environment variables.

    Returns a SimpleNamespace object allowing attribute access (e.g., config.budget)
    instead of dict-style access (config['budget']).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Expand environment variables in all config values
    config = expand_env_vars(config)

    # Convert to SimpleNamespace for attribute-style access
    return SimpleNamespace(**config)
