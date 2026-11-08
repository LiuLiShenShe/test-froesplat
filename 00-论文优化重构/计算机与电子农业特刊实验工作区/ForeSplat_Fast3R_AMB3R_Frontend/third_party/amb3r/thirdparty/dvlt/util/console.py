# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utils for rich console."""

from io import StringIO

from rich.console import Console
from rich.table import Table


def render_table_to_text(table: Table) -> str:
    """
    Render a rich Table to plain text suitable for standard log files.

    Rich tables use box-drawing characters that render fine in terminals and
    text files. This returns the textual representation so it can be logged
    via standard logging sinks (e.g., file handlers).
    """
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=False, color_system=None, force_jupyter=False)
    console.print(table)
    return buffer.getvalue().rstrip("\n")
