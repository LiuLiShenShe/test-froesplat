"""Minimal runtime compatibility for SAM3's resource_filename call."""

from __future__ import annotations

import importlib
from importlib import resources


def resource_filename(package: str, resource_name: str) -> str:
    module = importlib.import_module(package)
    resource = resources.files(module).joinpath(*resource_name.split("/"))
    return str(resource)
