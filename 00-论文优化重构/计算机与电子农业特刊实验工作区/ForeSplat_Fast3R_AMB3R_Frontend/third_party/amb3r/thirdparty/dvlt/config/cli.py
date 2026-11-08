# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
"""CLI utilities that bypass Hydra's job runner while keeping CLI functionality."""

import argparse
import difflib
import functools
import inspect
import os
import re
import sys
from typing import Any, Callable, Optional

from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

from .completion import BashCompletion


def _flatten_config_keys(cfg, prefix=""):
    """Extract all valid configuration keys from a resolved config."""
    keys = []
    if hasattr(cfg, "_metadata"):
        # This is an OmegaConf DictConfig
        for key, value in cfg.items():
            current_key = f"{prefix}.{key}" if prefix else key
            keys.append(current_key)
            # Recursively get nested keys
            if OmegaConf.is_config(value) and not OmegaConf.is_list(value):
                keys.extend(_flatten_config_keys(value, current_key))
    elif isinstance(cfg, dict):
        # Regular dictionary
        for key, value in cfg.items():
            current_key = f"{prefix}.{key}" if prefix else key
            keys.append(current_key)
            if isinstance(value, dict):
                keys.extend(_flatten_config_keys(value, current_key))
    return keys


def _extract_failed_override_key(error_message):
    """Extract the problematic override key from Hydra's error message."""
    # Look for patterns like "Could not override 'key.name'"
    match = re.search(r"Could not override ['\"]([^'\"]+)['\"]", str(error_message))
    if match:
        return match.group(1)
    return None


def _compose_config(cfg_dir, config_name, overrides, version_base, additional_paths=None):
    """Compose Hydra configuration with proper initialization context.

    Args:
        cfg_dir: Primary config directory
        config_name: Name of the config file to compose
        overrides: List of Hydra overrides
        version_base: Hydra version base
        additional_paths: List of additional config directories to add to search path
    """
    # If we have additional paths, add them via hydra.searchpath override
    if additional_paths:
        # Convert paths to absolute file:// URIs for Hydra's searchpath
        search_path_entries = []
        for add_path in additional_paths:
            abs_path = os.path.abspath(add_path) if not os.path.isabs(add_path) else add_path
            search_path_entries.append(f"file://{abs_path}")

        # Prepend the searchpath override to the overrides list
        searchpath_override = f"hydra.searchpath=[{','.join(search_path_entries)}]"
        overrides = [searchpath_override] + overrides

    # Standard Hydra initialization
    if os.path.isabs(cfg_dir):
        with initialize_config_dir(version_base=version_base, config_dir=cfg_dir):
            return compose(
                config_name=config_name,
                overrides=overrides,
                return_hydra_config=False,
            )
    else:
        with initialize(version_base=version_base, config_path=cfg_dir):
            return compose(
                config_name=config_name,
                overrides=overrides,
                return_hydra_config=False,
            )


def _suggest_similar_keys(failed_key, valid_keys, max_suggestions=3):
    """Find similar valid keys using string similarity."""
    if not failed_key:
        return []

    # Use difflib to find close matches
    close_matches = difflib.get_close_matches(failed_key, valid_keys, n=max_suggestions, cutoff=0.6)

    # Also look for keys that contain similar parts
    failed_parts = failed_key.split(".")
    if len(failed_parts) > 1:
        # Try matching just the last part (e.g., 'experiment_loggers' from 'trainer.experiment_loggers')
        last_part = failed_parts[-1]
        parent_prefix = ".".join(failed_parts[:-1])

        # Find keys under the same parent that are similar to the last part
        parent_keys = [key for key in valid_keys if key.startswith(parent_prefix + ".")]
        if parent_keys:
            last_parts = [key.split(".")[-1] for key in parent_keys]
            similar_last_parts = difflib.get_close_matches(last_part, last_parts, n=max_suggestions, cutoff=0.5)
            # Reconstruct full keys
            additional_matches = [f"{parent_prefix}.{part}" for part in similar_last_parts]
            close_matches.extend(additional_matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in close_matches:
        if match not in seen:
            seen.add(match)
            unique_matches.append(match)

    return unique_matches[:max_suggestions]


def cli(
    config_path: str,
    config_name: str,
    version_base: Optional[str] = None,
    extra_args: Optional[list[tuple[str, Any]]] = None,
) -> Callable:
    """Decorator that gives a function Hydra-style CLI behaviour without using the job runner.

    Args:
        config_path (str): Path to the directory that contains your Hydra config files.
        config_name (str): Name of the main config file (without extension).
        version_base (Optional[str]): Passed on to ``hydra.initialize``. Defaults to None.
        extra_args (Optional[list[tuple[str, Any]]]): Extra arguments to pass to the CLI.
            These will be parsed and passed to the wrapped function as keyword arguments.
    Returns:
        Callable: The decorated function with Hydra CLI functionality.

    The generated CLI supports these things:
        - Standard *help* for the CLI itself (``-h``/``--help``)
        - Showing the fully resolved configuration (``--help-config``)
        - Loading a custom config file via (``--config-path``) and (``--config-name``)
        - Regular Hydra override syntax (``key=value`` or ``group.key=value``)
        - Bash completion support (``-sc install``, ``-sc uninstall``)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:  # noqa: D401
            # Create bash completion handler
            completion = BashCompletion(config_path, config_name, version_base)

            # ------------------------------------------------------------------
            # 1. Parse *only* the arguments we care about (leave the rest as Hydra overrides)
            # ------------------------------------------------------------------
            parser = argparse.ArgumentParser(
                prog=sys.argv[0],
                description=func.__doc__,
                add_help=False,
                formatter_class=argparse.RawTextHelpFormatter,
            )
            parser.add_argument(
                "--config-path",
                type=str,
                default=None,
                help="Path to config directory (default search path will still be available for composition)",
            )
            parser.add_argument("--config-name", type=str, default=config_name, help="Name of config file")
            parser.add_argument("-sc", "--shell-completion", type=str, help="Shell completion command")
            for arg_name, arg_attrs in extra_args or []:
                parser.add_argument(arg_name, **arg_attrs)
            parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
            parser.add_argument(
                "--help-config",
                action="store_true",
                help="Print the resolved Hydra configuration and exit",
            )

            # Parse known args; everything else is treated as a Hydra override
            known, overrides = parser.parse_known_args()
            parsed_extra_args = {
                k: v
                for k, v in known.__dict__.items()
                if k not in ["config_path", "config_name", "help", "help_config", "shell_completion"]
            }

            # ------------------------------------------------------------------
            # 2. Handle shell completion commands
            # ------------------------------------------------------------------
            if known.shell_completion:
                cmd = known.shell_completion.lower()
                if cmd == "install":
                    completion.install()
                    return None
                elif cmd == "uninstall":
                    completion.uninstall()
                    return None
                elif cmd == "query":
                    suggestions = completion.query()
                    print(" ".join(suggestions))
                    return None
                else:
                    print(f"Unknown shell completion command: {cmd}")
                    return None

            # ------------------------------------------------------------------
            # 3. Handle CLI help
            # ------------------------------------------------------------------
            if known.help:
                parser.print_help()
                print(
                    "\nPositional CONFIG_OVERRIDES are forwarded to Hydra, e.g.:\n"
                    "  example_module=abc other_module.example_key=value \n"
                    "You can mix them with the options above.\n"
                    "\nBash completion can be installed with:\n"
                    '  eval "$(<cmd> -sc install)"'
                )
                return None

            # ------------------------------------------------------------------
            # 4. Resolve final config_path (handling relative path semantics)
            # ------------------------------------------------------------------
            # Did the user supply --config-path explicitly?
            user_supplied_path = known.config_path is not None

            # Resolve the default config path from the decorator
            if not os.path.isabs(config_path):
                script_dir = os.path.dirname(inspect.getfile(func))
                default_cfg_dir = os.path.abspath(os.path.join(script_dir, config_path))
            else:
                default_cfg_dir = config_path

            # Determine primary and additional config paths
            additional_paths = []
            if user_supplied_path:
                # User provided --config-path: use it as primary, add default to search path
                cfg_dir = (
                    os.path.abspath(os.path.join(os.getcwd(), known.config_path))
                    if not os.path.isabs(known.config_path)
                    else known.config_path
                )
                # Add the default path to additional search paths
                additional_paths.append(default_cfg_dir)
            else:
                # No --config-path provided: use decorator default as primary
                cfg_dir = default_cfg_dir

            # ------------------------------------------------------------------
            # 5. Compose the Hydra config (choose API depending on path type)
            # ------------------------------------------------------------------
            try:
                cfg = _compose_config(cfg_dir, known.config_name, overrides, version_base, additional_paths)
            except Exception as exc:  # pragma: no cover
                # Failed to compose configuration – show useful information then exit.
                print("[cli] Error while composing the configuration:\n", exc)

                # Try to provide "did you mean" suggestions for override errors
                failed_key = _extract_failed_override_key(str(exc))
                if failed_key:
                    try:
                        # Try to compose config without overrides to get valid keys
                        base_cfg = _compose_config(cfg_dir, known.config_name, [], version_base, additional_paths)

                        # Extract all valid configuration keys
                        valid_keys = _flatten_config_keys(base_cfg)

                        # Find similar keys
                        suggestions = _suggest_similar_keys(failed_key, valid_keys)

                        if suggestions:
                            print("\nDid you mean one of these?")
                            for suggestion in suggestions:
                                print(f"  {suggestion}")

                    except Exception:
                        # If we can't get suggestions, just continue with basic error
                        pass

                print("Run with -h/--help for usage information.")
                return None

            # ------------------------------------------------------------------
            # 6. If --help-config was requested, pretty-print the resolved config and exit
            # ------------------------------------------------------------------
            if known.help_config:
                print("# Resolved Hydra configuration\n")
                print(OmegaConf.to_yaml(cfg, resolve=True))
                return None

            # ------------------------------------------------------------------
            # 7. Normal execution – pass the Hydra config to the wrapped function
            # ------------------------------------------------------------------
            return func(cfg, *args, **parsed_extra_args, **kwargs)

        return wrapper

    return decorator
