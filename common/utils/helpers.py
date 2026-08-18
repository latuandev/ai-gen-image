import os
import tempfile
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


def env_bool(name: str, default: bool = False):
    """
    Get an environment variable and convert it to a boolean.
    """
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_list(name: str, default: str = ""):
    """
    Get a comma-separated environment variable and convert it to a list.
    """
    value = os.getenv(name, default)

    return [item.strip() for item in value.split(",") if item.strip()]


def default_agent_workspace_root() -> Path:
    """
    Return the canonical default local Agent workspace root.
    """

    return Path(tempfile.gettempdir()).resolve(strict=False) / "ai-gen-image" / "workspaces"


def agent_workspace_root_from_env(name: str = "AGENT_WORKSPACE_ROOT") -> Path:
    """
    Return the configured Agent workspace root or the canonical local default.

    Explicit operator configuration is returned as provided so filesystem
    safety policy is enforced by the WorkspaceManager.
    """

    configured_workspace_root = os.getenv(name)

    if configured_workspace_root is None or configured_workspace_root.strip() == "":
        return default_agent_workspace_root()

    workspace_root = Path(configured_workspace_root)

    if not workspace_root.is_absolute():
        raise ImproperlyConfigured(f"{name} must be an absolute path when set.")

    return workspace_root
