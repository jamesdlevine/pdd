#!/usr/bin/env python3
import sys
import argparse
import math
from typing import List, Tuple, Optional
from datetime import datetime

# Assuming the models are available in the python path as specified
try:
    from pdd.context_map_models import ContextMap
except ImportError:
    print("Error: Could not import 'pdd.context_map_models'. Ensure the package is installed or in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# --- Formatting Helpers ---

def format_chars(count: int) -> str:
    """Formats numbers with K suffix for thousands."""
    if count >= 1000:
        return f"{count/1000:.1f}K"
    return str(count)

def format_pct(part: int, total: int) -> str:
    """Formats percentage as integer string."""
    if total == 0:
        return "0%"
    return f"{int((part / total) * 100)}%"

def draw_bar(label: str, value: int, total: int, width: int = 40, suffix: str = "") -> str:
    """Draws an ASCII bar chart line."""
    if total == 0:
        filled_len = 0
        pct_str = "0%"
    else:
        pct = value / total
        filled_len = int(width * pct)
        pct_str = f"{int(pct * 100)}%"
    
    bar = "█" * filled_len + "░" * (width - filled_len)
    val_fmt = format_chars(value)
    return f"{label:<25} {bar} {val_fmt:>6} ({pct_str:>3}){suffix}"

# --- Visualization Logic ---

def print_header(context: ContextMap):
    """Prints the provenance header."""
    prov = context.provenance
    
    # Calculate duration in seconds
    duration_str = "N/A"
    if prov.duration_ms is not None:
        duration_str = f"{prov.duration_ms / 1000:.2f}s"

    # Format timestamp
    try:
        dt = datetime.fromisoformat(prov.timestamp_utc.replace('Z', '+00:00'))
        ts_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        ts_str = prov.timestamp_utc

    print(f"\nCONTEXT MAP REPORT")
    print(f"==================")
    print(f"Model:       {prov.model} ({prov.provider})")
    print(f"Prompt File: {prov.prompt_file}")
    print(f"Timestamp:   {ts_str}")
    print(f"Duration:    {duration_str}")
    if prov.pdd_version:
        print(f"PDD Version: {prov.pdd_version}")
    print(f"Total Input: {format_chars(context.input.total_chars)} chars")
    print("")

def render_summary_grid(context: ContextMap):
    """Renders the 10x10 summary grid."""
    inp = context.input
    total = inp.total_chars
    
    if total == 0:
        print("No input data to visualize.")
        return

    # Extract components safely using model defaults
    pb = inp.prompt_breakdown
    
    # Define categories with (Symbol, Label, Count)
    # Order: PDD system → devunit → includes → web → shell → variables → few-shot → prepend/append
    
    # 1. PDD System Prompt
    pdd_sys = pb.pdd_system_prompt_chars if pb else 0
    
    # 2. Devunit Prompt
    dev_prompt = pb.devunit_prompt_chars if pb else 0
    
    # 3-6. Preprocessor items
    inc_chars = pb.preprocessor_summary.include_chars if pb and pb.preprocessor_summary else 0
    web_chars = pb.preprocessor_summary.web_chars if pb and pb.preprocessor_summary else 0
    shell_chars = pb.preprocessor_summary.shell_chars if pb and pb.preprocessor_summary else 0
    var_chars = pb.preprocessor_summary.variable_chars if pb and pb.preprocessor_summary else 0
    
    # 7. Few Shot
    few_shot = pb.few_shot_total_chars if pb else 0
    
    # 8. Prepend/Append
    prep_app = (pb.prepended_chars + pb.appended_chars) if pb else 0

    # Calculate "Other/Unaccounted" to ensure grid fills 100 cells if math is slightly off due to structure
    # or if there are chars in api_structure not captured in breakdown
    known_sum = pdd_sys + dev_prompt + inc_chars + web_chars + shell_chars + var_chars + few_shot + prep_app
    remainder = max(0, total - known_sum)
    
    categories: List[Tuple[str, str, int]] = [
        ("▣", "PDD System Prompt", pdd_sys),
        ("◆", "Devunit Prompt", dev_prompt),
        ("█", "Disk Includes", inc_chars),
        ("▓", "Web Includes", web_chars),
        ("⌘", "Shell Output", shell_chars),
        ("•", "Variables", var_chars),
        ("◇", "Few-Shot Examples", few_shot),
        ("▤", "Prepend/Append", prep_app),
    ]
    
    if remainder > 0:
        categories.append(("?", "Other/Structure", remainder))

    # Calculate grid cells (total 100)
    grid_cells = []
    cells_remaining = 100
    
    # First pass: floor division
    counts = []
    for sym, label, count in categories:
        num_cells = math.floor((count / total) * 100)
        counts.append({"sym": sym, "label": label, "count": count, "cells": num_cells, "raw_frac": (count/total)*100})
        cells_remaining -= num_cells
    
    # Second pass: distribute remaining cells to largest fractional remainders
    counts.sort(key=lambda x: x["raw_frac"] - x["cells"], reverse=True)
    for i in range(cells_remaining):
        counts[i]["cells"] += 1
        
    # Restore original order for legend/drawing
    # We need to map back to the original order. 
    # Since the list is small, we can just rebuild the grid string based on the original categories list
    # and look up the cell count from the sorted 'counts' list.
    
    final_grid_str = ""
    legend_data = []
    
    for cat in categories:
        sym, label, count = cat
        # Find the calculated cell count
        cell_data = next(c for c in counts if c["label"] == label)
        num_cells = cell_data["cells"]
        final_grid_str += sym * num_cells
        legend_data.append((sym, label, count))

    # Print Grid (10 chars wide)
    print("Input Composition (1 cell ≈ 1%)")
    print("┌──────────┐")
    for i in range(0, 100, 10):
        row = final_grid_str[i:i+10]
        # Pad if short (shouldn't happen with logic above, but for safety)
        print(f"│{row:<10}│")
    print("└──────────┘")
    
    print("\nLegend:")
    for sym, label, count in legend_data:
        if count > 0:
            print(f" {sym} {label:<20} {format_chars(count):>6} ({format_pct(count, total)})")

def render_detailed_view(context: ContextMap):
    """Renders the detailed bar chart breakdown."""
    inp = context.input
    out = context.output
    
    # 1. Input / Output Overview
    print("INPUT / OUTPUT OVERVIEW")
    print("-" * 60)
    total_io = inp.total_chars + out.response_chars
    print(draw_bar("Input Chars", inp.total_chars, total_io))
    print(draw_bar("Output Chars", out.response_chars, total_io))
    print("")

    # 2. Tokens (if available)
    if out.prompt_tokens_reported or out.response_tokens_reported:
        print("TOKENS (Reported by Provider)")
        print("-" * 60)
        p_tok = out.prompt_tokens_reported or 0
        r_tok = out.response_tokens_reported or 0
        total_tok = p_tok + r_tok
        print(draw_bar("Prompt Tokens", p_tok, total_tok))
        print(draw_bar("Response Tokens", r_tok, total_tok))
        if out.response_tokens_estimated:
             print(f" (Estimated Response: {out.response_tokens_estimated})")
        print("")

    # 3. API Structure
    if inp.api_structure:
        api = inp.api_structure
        print("API STRUCTURE")
        print("-" * 60)
        # Total for API structure might differ slightly from total_chars due to overhead, use sum of parts
        api_total = api.system_prompt_chars + api.user_message_chars + api.assistant_prefill_chars + api.other_chars
        if api_total > 0:
            print(draw_bar("System Prompt", api.system_prompt_chars, api_total))
            print(draw_bar("User Message", api.user_message_chars, api_total))
            if api.assistant_prefill_chars > 0:
                print(draw_bar("Assistant Prefill", api.assistant_prefill_chars, api_total))
            if api.other_chars > 0:
                print(draw_bar("Other/Overhead", api.other_chars, api_total))
        print("")

    # 4. Prompt Breakdown
    if inp.prompt_breakdown:
        pb = inp.prompt_breakdown
        print("PROMPT BREAKDOWN")
        print("-" * 60)
        # Base components relative to total input
        print(draw_bar("PDD System Prompt", pb.pdd_system_prompt_chars, inp.total_chars))
        print(draw_bar("Devunit Prompt", pb.devunit_prompt_chars, inp.total_chars))
        print(draw_bar("Prepend/Append", pb.prepended_chars + pb.appended_chars, inp.total_chars))
        print(draw_bar("Preprocessor Total", pb.preprocessor_total_chars, inp.total_chars))
        print(draw_bar("Few-Shot Total", pb.few_shot_total_chars, inp.total_chars))
        print("")

        # 5. Preprocessor Details
        if pb.preprocessor_summary:
            ps = pb.preprocessor_summary
            print("PREPROCESSOR CONTENT")
            print("-" * 60)
            # Calculate total preprocessor chars for relative bars
            pp_total = ps.include_chars + ps.shell_chars + ps.web_chars + ps.variable_chars
            
            if pp_total > 0:
                print(draw_bar("File Includes", ps.include_chars, pp_total, suffix=f" [{ps.include_count} items]"))
                print(draw_bar("Web Includes", ps.web_chars, pp_total, suffix=f" [{ps.web_count} items]"))
                print(draw_bar("Shell Output", ps.shell_chars, pp_total, suffix=f" [{ps.shell_count} items]"))
                print(draw_bar("Variables", ps.variable_chars, pp_total, suffix=f" [{ps.variable_count} items]"))
            else:
                print("No preprocessor content.")
            print("")

        # 6. Few Shot Details
        if pb.few_shot_examples:
            print("FEW-SHOT EXAMPLES")
            print("-" * 60)
            # Sort by chars descending
            sorted_examples = sorted(pb.few_shot_examples, key=lambda x: x.chars, reverse=True)
            
            print(f"{'ID':<30} {'Size':<10} {'Pinned':<8} {'Score'}")
            print(f"{'-'*30} {'-'*10} {'-'*8} {'-'*5}")
            
            for ex in sorted_examples:
                pinned_mark = "YES" if ex.pinned else "-"
                score_str = f"{ex.quality_score:.2f}" if ex.quality_score is not None else "-"
                print(f"{ex.example_id[:28]:<30} {format_chars(ex.chars):<10} {pinned_mark:<8} {score_str}")
            print("")

def main():
    parser = argparse.ArgumentParser(description="Visualize PDD context map JSON files.")
    parser.add_argument("file", nargs="?", help="JSON context map file (reads stdin if omitted)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown instead of summary grid")
    
    args = parser.parse_args()

    # Load Data
    try:
        if args.file:
            context = ContextMap.from_file(args.file)
        else:
            if sys.stdin.isatty():
                parser.print_help()
                sys.exit(1)
            context = ContextMap.from_json(sys.stdin.read())
    except Exception as e:
        print(f"Error loading context map: {e}", file=sys.stderr)
        sys.exit(1)

    # Render
    print_header(context)
    
    if args.detailed:
        render_detailed_view(context)
    else:
        render_summary_grid(context)

if __name__ == "__main__":
    main()