#!/usr/bin/env python3
"""Generate text visualization of PDD context maps."""

import json
import sys
from typing import Any

# Grid symbols
SYM_FILLED = "⛁"
SYM_HALF = "⛀"
SYM_EMPTY = "⛶"


def format_chars(chars: int) -> str:
    """Format character count with K suffix for readability."""
    if chars >= 1000:
        return f"{chars / 1000:.1f}K"
    return str(chars)


def format_pct(chars: int, total: int) -> str:
    """Format as percentage."""
    if total == 0:
        return "0%"
    return f"{(chars / total) * 100:.0f}%"


def render_bar(chars: int, total: int, width: int = 30) -> str:
    """Render a proportional bar."""
    if total == 0:
        return ""
    filled = int((chars / total) * width)
    return "█" * filled + "░" * (width - filled)


def render_grid(segments: list[tuple[str, int]], total: int, grid_size: int = 100) -> list[str]:
    """Render a grid of symbols representing proportions.

    Args:
        segments: List of (symbol, char_count) tuples
        total: Total chars for calculating proportions
        grid_size: Total cells in the grid (default 100 = 10x10)

    Returns:
        List of strings, each representing one row of the grid
    """
    # Calculate cells per segment
    cells = []
    for sym, chars in segments:
        count = round((chars / total) * grid_size) if total > 0 else 0
        cells.extend([sym] * count)

    # Pad or trim to exact grid size
    while len(cells) < grid_size:
        cells.append(SYM_EMPTY)
    cells = cells[:grid_size]

    # Split into rows of 10
    rows = []
    for i in range(0, grid_size, 10):
        rows.append(" ".join(cells[i:i + 10]))
    return rows


def render_summary(context: dict[str, Any]) -> str:
    """Render a compact grid-based summary visualization."""
    lines = []

    prov = context.get("provenance", {})
    inp = context.get("input", {})
    out = context.get("output", {})
    breakdown = inp.get("prompt_breakdown", {})
    summary = breakdown.get("preprocessor_summary", {})

    total_input = inp.get("total_chars", 0)
    total = total_input  # Summary view only shows input breakdown

    # Calculate segment sizes
    pdd_system_chars = breakdown.get("pdd_system_prompt_chars", 0)
    devunit_chars = breakdown.get("devunit_prompt_chars", 0)
    prepended = breakdown.get("prepended_chars", 0)
    appended = breakdown.get("appended_chars", 0)
    prepend_append_total = prepended + appended
    include_chars = summary.get("include_chars", 0)
    shell_chars = summary.get("shell_chars", 0)
    web_chars = summary.get("web_chars", 0)
    variable_chars = summary.get("variable_chars", 0)
    few_shot_chars = breakdown.get("few_shot_total_chars", 0)

    # Build segments for grid (symbol, chars)
    # Order: PDD system, devunit, includes, web, shell, variables, few-shot, prepend/append
    segments = [
        ("▣", pdd_system_chars),    # PDD system prompt
        ("◆", devunit_chars),       # Devunit prompt (bare)
        ("█", include_chars),       # Disk file includes
        ("▓", web_chars),           # Web includes
        ("⌘", shell_chars),         # Shell includes
        ("•", variable_chars),      # Variable substitutions
        ("◇", few_shot_chars),      # Few-shot examples
        ("▤", prepend_append_total),# Prepend/append (PDD system)
    ]

    grid_rows = render_grid(segments, total)

    # Header
    lines.append("PDD CONTEXT MAP")
    lines.append(f"{prov.get('prompt_file', 'unknown')} · {format_chars(total_input)} total input")
    lines.append("")

    # Build legend entries (matching grid order)
    few_shot_count = len(breakdown.get("few_shot_examples", []))
    legend = [
        f"▣ PDD system: {format_chars(pdd_system_chars)} ({format_pct(pdd_system_chars, total)})" if pdd_system_chars else None,
        f"◆ Devunit: {format_chars(devunit_chars)} ({format_pct(devunit_chars, total)})" if devunit_chars else None,
        f"█ Includes: {format_chars(include_chars)} ({format_pct(include_chars, total)})" if include_chars else None,
        f"▓ Web: {format_chars(web_chars)} ({format_pct(web_chars, total)})" if web_chars else None,
        f"⌘ Shell: {format_chars(shell_chars)} ({format_pct(shell_chars, total)})" if shell_chars else None,
        f"• Variables: {format_chars(variable_chars)} ({format_pct(variable_chars, total)})" if variable_chars else None,
        f"◇ Few-shot: {few_shot_count}x {format_chars(few_shot_chars)} ({format_pct(few_shot_chars, total)})" if few_shot_chars else None,
        f"▤ Prepend/append: {format_chars(prepend_append_total)} ({format_pct(prepend_append_total, total)})" if prepend_append_total else None,
    ]
    legend = [l for l in legend if l is not None]

    # Combine grid and legend side by side
    for i, row in enumerate(grid_rows):
        if i < len(legend):
            lines.append(f"{row}   {legend[i]}")
        else:
            lines.append(row)

    lines.append("")
    lines.append(f"Duration: {prov.get('duration_ms', 0)}ms · {prov.get('timestamp_utc', '')}")

    return "\n".join(lines)


def render_detailed(context: dict[str, Any]) -> str:
    """Render a detailed breakdown with bar charts."""
    lines = []

    # Header
    prov = context.get("provenance", {})
    lines.append("─" * 60)
    lines.append("PDD CONTEXT MAP")
    lines.append("─" * 60)

    # Provenance
    lines.append(f"Prompt:    {prov.get('prompt_file', 'unknown')}")
    lines.append(f"Model:     {prov.get('model', 'unknown')}")
    lines.append(f"Provider:  {prov.get('provider', 'unknown')}")
    lines.append(f"Timestamp: {prov.get('timestamp_utc', 'unknown')}")
    lines.append(f"Duration:  {prov.get('duration_ms', 0)}ms")
    lines.append("")

    # Input/Output summary
    inp = context.get("input", {})
    out = context.get("output", {})
    total_input = inp.get("total_chars", 0)
    total_output = out.get("response_chars", 0)
    total = total_input + total_output

    # Prompt breakdown
    breakdown = inp.get("prompt_breakdown", {})

    # PDD system prompt
    pdd_system_chars = breakdown.get("pdd_system_prompt_chars", 0)
    if pdd_system_chars:
        lines.append("PDD SYSTEM PROMPT")
        lines.append(f"  {format_chars(pdd_system_chars)}")
        lines.append("")

    lines.append("INPUT/OUTPUT")
    lines.append(f"  Input:  {render_bar(total_input, total)} {format_chars(total_input)}")
    lines.append(f"  Output: {render_bar(total_output, total)} {format_chars(total_output)}")
    lines.append("")

    # API structure breakdown
    api = inp.get("api_structure", {})
    if api:
        lines.append("API STRUCTURE")
        for key in ["system_prompt_chars", "user_message_chars", "assistant_prefill_chars", "other_chars"]:
            chars = api.get(key, 0)
            if chars > 0:
                label = key.replace("_chars", "").replace("_", " ").title()
                lines.append(f"  {label:20} {render_bar(chars, total_input, 25)} {format_chars(chars)}")
        lines.append("")

    # Prompt breakdown
    if breakdown:
        lines.append("PROMPT BREAKDOWN")
        devunit = breakdown.get("devunit_prompt_chars", 0)
        prepended = breakdown.get("prepended_chars", 0)
        appended = breakdown.get("appended_chars", 0)
        preproc = breakdown.get("preprocessor_total_chars", 0)
        prompt_total = devunit + prepended + appended + preproc

        lines.append(f"  {'Devunit prompt':20} {render_bar(devunit, prompt_total, 25)} {format_chars(devunit)}")
        if prepended:
            lines.append(f"  {'Prepended':20} {render_bar(prepended, prompt_total, 25)} {format_chars(prepended)}")
        if appended:
            lines.append(f"  {'Appended':20} {render_bar(appended, prompt_total, 25)} {format_chars(appended)}")
        lines.append(f"  {'Preprocessor':20} {render_bar(preproc, prompt_total, 25)} {format_chars(preproc)}")
        lines.append("")

    # Preprocessor summary by type
    summary = breakdown.get("preprocessor_summary", {})
    if summary:
        lines.append("PREPROCESSOR")
        preproc_total = breakdown.get("preprocessor_total_chars", 1)
        type_order = ["include", "shell", "web", "variable"]
        for t in type_order:
            count = summary.get(f"{t}_count", 0)
            chars = summary.get(f"{t}_chars", 0)
            if count > 0:
                lines.append(f"  {t.title():12} {render_bar(chars, preproc_total, 20)} {count:2}x {format_chars(chars):>8}")
        lines.append("")

    # Few-shot examples
    few_shot = breakdown.get("few_shot_examples", [])
    if few_shot:
        few_shot_total = breakdown.get("few_shot_total_chars", 1)
        lines.append("FEW-SHOT EXAMPLES")
        for ex in few_shot:
            example_id = ex.get("example_id", "unknown")[:8]  # Show first 8 chars of UUID
            chars = ex.get("chars", 0)
            pinned = "📌" if ex.get("pinned", False) else "  "
            quality = ex.get("quality_score")
            quality_str = f"q={quality:.2f}" if quality is not None else ""
            lines.append(f"  {pinned} {example_id}...  {render_bar(chars, few_shot_total, 15)} {format_chars(chars):>6} {quality_str}")
        lines.append(f"     {'Total':12} {' ' * 15} {format_chars(few_shot_total):>6}")
        lines.append("")

    lines.append("─" * 60)
    return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PDD context maps")
    parser.add_argument("file", nargs="?", help="JSON context map file (or stdin)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown instead of summary")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            context = json.load(f)
    else:
        context = json.load(sys.stdin)

    if args.detailed:
        print(render_detailed(context))
    else:
        print(render_summary(context))


if __name__ == "__main__":
    main()
