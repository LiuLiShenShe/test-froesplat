# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Bash completion support for custom CLI."""

import ast
import importlib.util
import os
import sys
from typing import Any, Optional

from hydra import compose, initialize, initialize_config_dir


class BashCompletion:
    """Bash completion support for custom CLI."""

    def __init__(self, config_path: str, config_name: str, version_base: Optional[str] = None):
        self.config_path = config_path
        self.config_name = config_name
        self.version_base = version_base

    def install(self) -> None:
        """Generate bash completion installation script."""
        script = f"export _HYDRA_OLD_COMP=$(complete -p {self._get_exec()} 2> /dev/null)\n"
        script += r"""hydra_bash_completion() {
    words=($COMP_LINE)
    if [ "${words[0]}" == "python" ]; then
        if (( ${#words[@]} < 2 )); then
            return
        fi

        # Handle both "python script.py" and "python -m module.name" syntax
        if [ "${words[1]}" == "-m" ]; then
            if (( ${#words[@]} < 3 )); then
                return
            fi
            # For python -m module.name, try it and let the -sc query determine if it's compatible
            helper="${words[0]} ${words[1]} ${words[2]}"
        else
            # Handle "python script.py" syntax
            file_path=$(pwd)/${words[1]}
            if [ ! -f "$file_path" ]; then
                return
            fi
            # For python script.py syntax, we'll attempt completion and let the -sc query determine compatibility
            # This allows the script itself to determine if it supports shell completion
            helper="${words[0]} ${words[1]}"
        fi
    else
        helper="${words[0]}"
    fi

    EXECUTABLE=($(command -v $helper))
    if [ "$HYDRA_COMP_DEBUG" == "1" ]; then
        printf "EXECUTABLE_FIRST='${EXECUTABLE[0]}'\\n"
    fi

    if ! [ -x "${EXECUTABLE[0]}" ]; then
        return
    fi

    choices=$(
        COMP_POINT=$COMP_POINT COMP_LINE="$COMP_LINE" $helper -sc query 2>/dev/null
    )
    word=${words[$COMP_CWORD]}

    if [ "$HYDRA_COMP_DEBUG" == "1" ]; then
        printf "\\n"
        printf "COMP_LINE='$COMP_LINE'\\n"
        printf "COMP_POINT='$COMP_POINT'\\n"
        printf "Word='$word'\\n"
        printf "Output suggestions:\\n"
        printf "\\t%s\\n" ${choices[@]}
    fi

    COMPREPLY=($(
        compgen -o nospace -o default -W "$choices" -- "$word"
    ))
}

COMP_WORDBREAKS=${COMP_WORDBREAKS//=}
complete -o nospace -o default -F hydra_bash_completion """
        print(script + self._get_exec())

    def uninstall(self) -> None:
        """Generate bash completion uninstallation script."""
        print("unset hydra_bash_completion")
        print(os.environ.get("_HYDRA_OLD_COMP", ""))
        print("unset _HYDRA_OLD_COMP")

    def query(self) -> list[str]:
        """Query available completions based on current command line."""
        line = os.environ.get("COMP_LINE", "")
        if not line:
            return []

        # Strip the executable name from the line
        line = self._strip_exec_name(line)

        # Get completion suggestions
        return self._get_completions(line)

    @staticmethod
    def help(command: str) -> str:
        """Generate help text for bash completion commands."""
        assert command in ["install", "uninstall"]
        return f'eval "$({sys.argv[0]} -sc {command})"'

    def _get_completions(self, line: str) -> list[str]:
        """Get completion suggestions for the given command line."""
        words = line.strip().split()
        if not words:
            return self._get_base_completions()

        last_word = words[-1] if words else ""

        # If the last word looks like a key=value assignment, complete the value
        if "=" in last_word:
            return self._complete_override_value(last_word)

        # If the last word starts with --, complete CLI flags
        if last_word.startswith("--"):
            return self._complete_cli_flags(last_word)

        # Otherwise, complete config keys and groups
        return self._complete_config_keys(last_word)

    def _get_base_completions(self) -> list[str]:
        """Get base completion options (CLI flags + config keys)."""
        completions = []

        # Add CLI flags
        completions.extend(["--config-path", "--config-name", "--help", "--help-config"])

        # Add config keys from the default config
        try:
            completions.extend(self._get_config_keys())
        except Exception:
            pass  # Ignore errors when getting config keys

        return completions

    def _complete_cli_flags(self, prefix: str) -> list[str]:
        """Complete CLI flags that start with the given prefix."""
        flags = ["--config-path", "--config-name", "--help", "--help-config"]
        return [flag for flag in flags if flag.startswith(prefix)]

    def _complete_config_keys(self, prefix: str) -> list[str]:
        """Complete config keys that start with the given prefix."""
        try:
            keys = self._get_config_keys()
            return [key for key in keys if key.startswith(prefix)]
        except Exception:
            return []

    def _complete_override_value(self, override: str) -> list[str]:
        """Complete values for config overrides (key=value)."""
        if "=" not in override:
            return []

        key, partial_value = override.split("=", 1)

        try:
            # Try to get possible values for this key from the schema
            possible_values = self._get_possible_values_for_key(key)
            return [f"{key}={val}" for val in possible_values if val.startswith(partial_value)]
        except Exception:
            return []

    def _extract_script_path_from_command_line(self) -> str | None:
        """Extract the script path from the current command line."""
        line = os.environ.get("COMP_LINE", "")
        if not line:
            return None

        words = line.split()
        if len(words) < 2:
            return None

        # Handle "python -m module.name" syntax
        if words[0] == "python" and len(words) >= 3 and words[1] == "-m":
            module_name = words[2]
            # Try to find the module file using standard Python module resolution
            # First try common project structures
            possible_module_paths = [
                f"src/{module_name.replace('.', '/')}.py",  # src/package/module.py
                f"{module_name.replace('.', '/')}.py",  # package/module.py
                f"lib/{module_name.replace('.', '/')}.py",  # lib/package/module.py
            ]

            for script_path in possible_module_paths:
                if os.path.exists(script_path):
                    return os.path.abspath(script_path)

        # Handle "python script.py" syntax
        elif words[0] == "python" and len(words) >= 2:
            script_path = words[1]
            if os.path.exists(script_path):
                return os.path.abspath(script_path)

        return None

    def _parse_cli_decorator(self, script_path: str) -> tuple[str, str, str | None]:
        """Parse the @cli decorator from a script to extract config_path, config_name, and version_base."""
        try:
            with open(script_path, "r") as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content)

            # Look for @cli decorator
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        # Handle @cli(...) calls
                        if (
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Name)
                            and decorator.func.id == "cli"
                        ):
                            config_path = None
                            config_name = None
                            version_base = None

                            # Extract keyword arguments
                            for keyword in decorator.keywords:
                                if keyword.arg == "config_path" and isinstance(keyword.value, ast.Constant):
                                    config_path = keyword.value.value
                                elif keyword.arg == "config_name" and isinstance(keyword.value, ast.Constant):
                                    config_name = keyword.value.value
                                elif keyword.arg == "version_base" and isinstance(keyword.value, ast.Constant):
                                    version_base = keyword.value.value

                            if config_path and config_name:
                                return config_path, config_name, version_base

            # Fallback to default values
            return self.config_path, self.config_name, self.version_base

        except Exception:
            # If parsing fails, use default values
            return self.config_path, self.config_name, self.version_base

    def _find_and_execute_config_registration(self, script_path: str) -> None:
        """Find and execute config registration functions from the script."""
        try:
            with open(script_path, "r") as f:
                content = f.read()

            # Parse the AST to find registration calls
            tree = ast.parse(content)

            # Look for function calls that might register configs
            registration_calls = []

            for node in ast.walk(tree):
                # Look for direct function calls like register_configs()
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == "register_configs":
                        registration_calls.append("register_configs")
                    elif isinstance(node.func, ast.Attribute) and node.func.attr == "store":
                        # Look for ConfigStore.store() calls
                        registration_calls.append("config_store")

            # Execute the script's imports and registration
            script_dir = os.path.dirname(script_path)
            script_name = os.path.basename(script_path)[:-3]  # Remove .py

            # Add script directory to path
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)

            try:
                # Try to import and execute the script's registration
                spec = None
                module = None

                # Try to determine module structure from script path
                # Look for common Python project structures
                possible_roots = []
                current_dir = os.path.dirname(script_path)

                # Walk up to find possible project roots (containing setup.py, pyproject.toml, or src/ dir)
                while current_dir != os.path.dirname(current_dir):  # Not at filesystem root
                    if any(
                        os.path.exists(os.path.join(current_dir, marker))
                        for marker in ["setup.py", "pyproject.toml", "src"]
                    ):
                        possible_roots.append(current_dir)
                    current_dir = os.path.dirname(current_dir)

                # Also try direct parent directories of common source roots
                for src_dir in ["src", "lib"]:
                    if f"/{src_dir}/" in script_path:
                        src_parent = script_path.split(f"/{src_dir}/")[0]
                        possible_roots.append(src_parent)

                # Try each possible root to create a module name
                for root in possible_roots:
                    try:
                        rel_path = os.path.relpath(script_path, root)
                        # Skip if it goes up from the root
                        if not rel_path.startswith(".."):
                            module_name = rel_path[:-3].replace("/", ".")  # Remove .py and convert slashes
                            break
                    except ValueError:
                        continue
                else:
                    # Fallback: use just the script name
                    module_name = script_name

                    spec = importlib.util.spec_from_file_location(module_name, script_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # Execute only the imports and registration, not the main function
                        with open(script_path, "r") as f:
                            lines = f.readlines()

                        # Execute only import lines and likely registration calls
                        import_and_reg_code = []
                        for line in lines:
                            stripped = line.strip()
                            # Import statements
                            if stripped.startswith("import ") or stripped.startswith("from "):
                                import_and_reg_code.append(line)
                            # Common config registration patterns (generic)
                            elif any(
                                pattern in stripped
                                for pattern in [
                                    "ConfigStore",
                                    "cs.store",
                                    "register_",
                                    ".store(",
                                    ".instance()",
                                    "CS =",
                                ]
                            ):
                                import_and_reg_code.append(line)

                        if import_and_reg_code:
                            exec_code = "".join(import_and_reg_code)
                            exec(exec_code, module.__dict__)

            except Exception:
                pass  # Ignore import errors

        except Exception:
            pass  # Ignore any parsing errors

    def _parse_current_command_line(self) -> tuple[list[str], str]:
        """Parse the current command line to extract overrides and config_name."""
        line = os.environ.get("COMP_LINE", "")
        if not line:
            return [], self.config_name

        # Strip the executable part
        stripped_line = self._strip_exec_name(line)
        words = stripped_line.split()

        config_name = self.config_name
        overrides = []

        i = 0
        while i < len(words):
            word = words[i]

            # Handle --config-name
            if word == "--config-name" and i + 1 < len(words):
                config_name = words[i + 1].replace(".yaml", "")  # Remove .yaml extension if present
                i += 2
            elif word.startswith("--config-name="):
                config_name = word.split("=", 1)[1].replace(".yaml", "")
                i += 1
            # Handle other CLI flags (skip them)
            elif word.startswith("--"):
                if "=" in word:
                    i += 1  # Skip flag=value
                elif i + 1 < len(words) and not words[i + 1].startswith("--"):
                    i += 2  # Skip flag and its value
                else:
                    i += 1  # Skip flag without value
            # Handle overrides (key=value)
            elif "=" in word and not word.startswith("-"):
                overrides.append(word)
                i += 1
            else:
                i += 1

        return overrides, config_name

    def _get_config_keys(self) -> list[str]:
        """Get all available config keys from the current configuration context."""
        try:
            # Extract script path from command line and parse its @cli decorator
            script_path = self._extract_script_path_from_command_line()

            if script_path:
                # Parse the script's @cli decorator to get the actual config settings
                config_path, config_name, version_base = self._parse_cli_decorator(script_path)

                # Execute the script's config registration
                self._find_and_execute_config_registration(script_path)
            else:
                # Fallback to instance defaults if no script path detected
                config_path = self.config_path
                config_name = self.config_name
                version_base = self.version_base
                # Note: No config registration in fallback - rely entirely on dynamic detection

            # Parse the current command line to extract overrides and config_name
            overrides, parsed_config_name = self._parse_current_command_line()

            # Use parsed config_name if available, otherwise use the one from @cli decorator
            final_config_name = parsed_config_name if parsed_config_name != self.config_name else config_name

            # Resolve config path using the same logic as the CLI decorator
            if os.path.isabs(config_path):
                # Absolute path - use as is
                cfg_dir = config_path
                ctx_manager = initialize_config_dir(version_base=version_base, config_dir=cfg_dir)
            # Relative path - resolve relative to the script's directory (same as CLI decorator)
            elif script_path:
                script_dir = os.path.dirname(script_path)
                cfg_dir = os.path.abspath(os.path.join(script_dir, config_path))

                if os.path.exists(cfg_dir):
                    ctx_manager = initialize_config_dir(version_base=version_base, config_dir=cfg_dir)
                else:
                    # If resolved path doesn't exist, fall back to Hydra's search mechanism
                    ctx_manager = initialize(version_base=version_base, config_path=config_path)
            else:
                # No script path available, use Hydra's search mechanism
                ctx_manager = initialize(version_base=version_base, config_path=config_path)

            with ctx_manager:
                # Use the parsed config_name and overrides for context-aware completion
                cfg = compose(config_name=final_config_name, overrides=overrides, return_hydra_config=False)
                return self._extract_keys_from_config(cfg)
        except Exception:
            return []

    def _extract_keys_from_config(self, cfg: Any, prefix: str = "") -> list[str]:
        """Recursively extract all keys from a config object."""
        keys = []

        if hasattr(cfg, "_content") and cfg._content:
            # Check if _content is a dictionary-like object
            if hasattr(cfg._content, "items"):
                for key, value in cfg._content.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    keys.append(full_key)

                    # If the value is also a config object, recurse
                    if hasattr(value, "_content") and hasattr(value._content, "items"):
                        keys.extend(self._extract_keys_from_config(value, full_key))
            # If _content is not dictionary-like (e.g., list, tuple), don't recurse

        return keys

    def _get_possible_values_for_key(self, key: str) -> list[str]:
        """Get possible values for a given config key."""
        # This is a simplified implementation
        # In a full implementation, you might introspect the schema
        # or config groups to provide better suggestions

        # For boolean keys, suggest true/false
        if key.lower().endswith(("_enabled", "_enable", "_flag", "_debug")):
            return ["true", "false"]

        # For path keys, could suggest file completions
        if "path" in key.lower() or "dir" in key.lower():
            return []  # File completion would be handled by bash itself

        return []

    def _strip_exec_name(self, line: str) -> str:
        """Strip the executable name from the command line."""
        words = line.split()
        if not words:
            return line

        # If first word is 'python', handle both syntaxes
        if words[0] == "python" and len(words) > 1:
            if len(words) > 2 and words[1] == "-m":
                # For "python -m module.name", remove first 3 words
                return " ".join(words[3:])
            else:
                # For "python script.py", remove first 2 words
                return " ".join(words[2:])
        else:
            return " ".join(words[1:])

    @staticmethod
    def _get_exec() -> str:
        """Get the executable name for this script."""
        if sys.argv[0].endswith(".py"):
            return "python"
        else:
            return os.path.basename(sys.argv[0])
