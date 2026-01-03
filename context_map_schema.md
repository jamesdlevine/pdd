# PDD Context Map Schema

## Overview

Schema design for logging context metadata during `pdd generate` for subsequent analysis and visualization.

## Requirements (Initial Description)

Capture a context map for each generation round with the following data:

- **Character counts** (not tokens, since token counts are not very accurate)
- **Total input size** to the LLM API call
- **API structure breakdown** (varies by provider): system prompt, user input, other inputs
- **Prompt breakdown**:
  - Base code file prompt size
  - Size of each included/prepended/appended text
- **Preprocessor artifacts** - individual instances with sizes:
  - File includes (`<include>` directive or triple-backtick syntax)
  - `<shell>` command outputs
  - `<web>` fetched content
  - Variable substitutions
- **Response size** (LLM-generated text, not full API response)
- **Provenance**: model called, date/time, duration of API call, etc.

## JSON Schema Example

```json
{
  "schema_version": "1.0",
  "generation_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",

  "pdd_system_prompt_chars": 3200,

  "provenance": {
    "timestamp_utc": "2026-01-01T14:30:00Z",
    "duration_ms": 2340,
    "model": "claude-sonnet-4-20250514",
    "provider": "anthropic",
    "prompt_file": "src/prompts/feature.prompt.md",
    "pdd_version": "0.5.2"
  },

  "input": {
    "total_chars": 48500,

    "api_structure": {
      "system_prompt_chars": 4200,
      "user_message_chars": 44300,
      "assistant_prefill_chars": 0,
      "other_chars": 0
    },

    "prompt_breakdown": {
      "base_prompt_chars": 1850,

      "prepended_chars": 2400,
      "appended_chars": 600,

      "preprocessor_total_chars": 39650,
      "preprocessor_items": [
        {
          "type": "include",
          "syntax": "directive",
          "source": "src/utils/helpers.py",
          "chars": 12400,
          "line_in_prompt": 5,
          "include_many": false
        },
        {
          "type": "include",
          "syntax": "backticks",
          "source": "src/models/user.py",
          "chars": 8200,
          "line_in_prompt": 12,
          "include_many": true
        },
        {
          "type": "shell",
          "command": "tree src/ -L 2",
          "chars": 1200,
          "line_in_prompt": 3
        },
        {
          "type": "web",
          "url": "https://docs.example.com/api",
          "chars": 15000,
          "line_in_prompt": 20
        },
        {
          "type": "variable",
          "name": "MODULE_NAME",
          "chars": 50,
          "line_in_prompt": 1
        }
      ],

      "preprocessor_summary": {
        "include_count": 2,
        "include_chars": 20600,
        "shell_count": 1,
        "shell_chars": 1200,
        "web_count": 1,
        "web_chars": 15000,
        "variable_count": 1,
        "variable_chars": 50
      },

      "few_shot_examples": [
        { "example_id": "f7e8d9c0-a1b2-4c3d-8e5f-6a7b8c9d0e1f", "chars": 2400, "pinned": true, "quality_score": 0.95 },
        { "example_id": "b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e", "chars": 1800, "pinned": false, "quality_score": 0.82 }
      ],
      "few_shot_total_chars": 4200
    }
  },

  "output": {
    "response_chars": 8750
  }
}
```

## Design Rationale

| Section | Purpose |
|---------|---------|
| `pdd_system_prompt_chars` | Size of PDD runtime prompt prepended to all generations |
| `provenance` | Who/what/when for reproducibility and cost attribution |
| `api_structure` | Maps to actual API call structure (varies by provider) |
| `base_prompt_chars` | The raw prompt file before preprocessing |
| `prepended/appended` | Config-driven injections (separate from inline directives) |
| `preprocessor_items` | Individual instances; `include_many` flag distinguishes bulk includes |
| `preprocessor_summary` | Aggregated stats for quick visualization (pie charts, etc.) |
| `few_shot_examples` | Example IDs, sizes, `pinned` status, and `quality_score` from cloud mode |

## CLI Visualization Tool

`context_viz.py` renders context maps as text visualizations.

### Usage

```bash
# Summary view (default) - grid-based visualization
python3 context_viz.py context.json

# Detailed view - full bar-chart breakdown
python3 context_viz.py context.json --detailed

# From stdin
cat context.json | python3 context_viz.py
```

### Summary Output

```
PDD CONTEXT MAP

▣ ▣ ▣ ▣ ▣ ▣ █ █ █ █   src/prompts/feature.prompt.md
█ █ █ █ █ █ █ █ █ █   claude-sonnet-4-20250514 · 48.5K in / 8.8K out
█ █ █ █ █ █ █ █ █ █
█ █ █ █ █ █ █ █ █ █   ▣ PDD system: 3.2K (6%)
█ █ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓   █ Includes: 20.6K (36%)
▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓   ▓ Web: 15.0K (26%)
▓ ▓ ▓ ▓ ▓ ▓ ▓ ▓ ▒ ▒   ▒ Shell: 1.2K (2%)
◇ ◇ ◇ ◇ ◇ ◇ ◇ ◆ ◆ ◆   ░ Variables: 50 (0%)
◆ ◆ ◆ ◆ ◆ ○ ○ ○ ○ ○   ◇ Few-shot: 2x 4.2K (7%)
○ ○ ○ ○ ○ ○ ○ ○ ○ ○   ◆ Base/prepend/append: 4.8K (8%)

Duration: 2340ms · 2026-01-01T14:30:00Z
```

### Detailed Output

```
────────────────────────────────────────────────────────────
PDD CONTEXT MAP
────────────────────────────────────────────────────────────
Prompt:    src/prompts/feature.prompt.md
Model:     claude-sonnet-4-20250514
Provider:  anthropic
Timestamp: 2026-01-01T14:30:00Z
Duration:  2340ms

PDD SYSTEM PROMPT
  3.2K

INPUT/OUTPUT
  Input:  █████████████████████████░░░░░ 48.5K
  Output: ████░░░░░░░░░░░░░░░░░░░░░░░░░░ 8.8K

API STRUCTURE
  System Prompt        ██░░░░░░░░░░░░░░░░░░░░░░░ 4.2K
  User Message         ██████████████████████░░░ 44.3K

PROMPT BREAKDOWN
  Base prompt          █░░░░░░░░░░░░░░░░░░░░░░░░ 1.9K
  Prepended            █░░░░░░░░░░░░░░░░░░░░░░░░ 2.4K
  Appended             ░░░░░░░░░░░░░░░░░░░░░░░░░ 600
  Preprocessor         ██████████████████████░░░ 39.6K

PREPROCESSOR
  Include      ██████████░░░░░░░░░░  2x    20.6K
  Shell        ░░░░░░░░░░░░░░░░░░░░  1x     1.2K
  Web          ███████░░░░░░░░░░░░░  1x    15.0K
  Variable     ░░░░░░░░░░░░░░░░░░░░  1x       50

FEW-SHOT EXAMPLES
  📌 f7e8d9c0...  ████████░░░░░░░   2.4K q=0.95
     b2c3d4e5...  ██████░░░░░░░░░   1.8K q=0.82
     Total                          4.2K

────────────────────────────────────────────────────────────
```

## Open Questions

1. **Multiple files per generation** - Does `sync` ever batch multiple prompts? Should this be an array?
2. **Error cases** - Capture failed generations (timeout, rate limit)?
3. **Cost data** - Include estimated cost alongside chars if available?
4. **Output file mapping** - Track which output files were written?
