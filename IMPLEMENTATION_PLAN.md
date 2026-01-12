# Context Map Implementation Plan for PDD

## Overview

Add context map capture to `pdd generate`/`pdd sync` to log detailed metrics about each generation for analysis and visualization.

**Approach**: This plan follows Prompt-Driven Development principles. All changes are made by modifying prompts, then regenerating code. Prompts are the source of truth; generated code is ephemeral.

## New Prompts to Create

### 1. `context_map_models_python.prompt` (NEW)

**Purpose**: Pydantic models matching the JSON schema. Single source of truth for data structures.

**Location**: `pdd/prompts/context_map_models_python.prompt`

**Key Requirements**:
- Define models strictly matching `context_map.schema.json`
- Export `ContextMap`, `Provenance`, `Input`, `Output`, nested models, and enums
- Provide `ContextMap.generate_sample()` for testing
- Convenience methods: `to_json()`, `from_file()`, `save()`

**Dependencies**: `context_map.schema.json` (via backtick include)

**Prompt-to-Code Ratio**: ~20% (schema drives structure)

---

### 2. `context_sampler_python.prompt` (NEW)

**Purpose**: Storage layer for context map data during generation.

**Location**: `pdd/prompts/context_sampler_python.prompt`

**Key Requirements**:
- Accept `ContextMap` instances and persist to JSON
- Store in `.pdd_context/` directory alongside output
- File naming: `<basename>.context.<N>.json` with monotonic N
- Retention: keep most recent N files (default 5)
- CLI `--sample` flag outputs example via `ContextMap.generate_sample()`

**Dependencies**: `context_map_models` (import models)

**Contracts**:
- Input: ContextMap instance
- Output: path to written file, or None on error
- Invariant: file count per devunit ≤ configured maximum

---

### 3. `context_viz_python.prompt` (NEW)

**Purpose**: CLI visualization of context map JSON files.

**Location**: `pdd/prompts/context_viz_python.prompt`

**Key Requirements**:
- Read JSON from file argument or stdin
- Two modes: summary (10x10 grid) and detailed (bar charts)
- Summary grid symbols: ▣ PDD system, ◆ devunit, █ includes, ▓ web, ⌘ shell, • variables, ◇ few-shot, ▤ prepend/append
- Show provenance (prompt file, model, timestamp, duration)

**Dependencies**: `context_map_models` (import `ContextMap`)

---

## Existing Prompts to Modify

### 4. `llm_invoke_python.prompt` (MODIFY)

**Goal**: Expose token counts and duration in return dict.

**Add to "Output (dictionary)" section**:
```
% Output (dictionary):
    - 'result': String or Pydantic object (or list if batch mode).
    - 'cost': Total cost via LiteLLM callback.
    - 'model_name': Name of the selected model.
    - 'thinking_output': Reasoning output if available.
    # NEW fields:
    - 'input_tokens': Prompt tokens from callback (0 if unavailable).
    - 'output_tokens': Completion tokens from callback (0 if unavailable).
    - 'duration_ms': API call duration in milliseconds.
    - 'provider': Provider name extracted from model info.
```

**Behavioral change**: Return dict includes timing and token metadata for downstream instrumentation.

---

### 5. `preprocess_python.prompt` (MODIFY)

**Goal**: Return preprocessing metadata alongside expanded text.

**Add new section**:
```
% Metadata Collection Mode

When called via `preprocess_with_metadata()`:
- Return tuple: (expanded_text, PreprocessMetadata)
- Track each directive processed: type, source/command/url/name, chars produced, line number
- Compute summary: counts and char totals by type (include, shell, web, variable)
- Original `preprocess()` remains unchanged for backward compatibility

% PreprocessMetadata Structure
- items: List of PreprocessorItem (type, source, chars, line_in_prompt, syntax, include_many)
- summary: Dict with {type}_count and {type}_chars for each preprocessor type
```

**New function signature**:
```python
def preprocess_with_metadata(prompt: str, ...) -> Tuple[str, PreprocessMetadata]
```

---

### 6. `code_generator_main_python.prompt` (MODIFY)

**Goal**: Instrument preprocessing and record metrics for both cloud and local paths.

**Add new section**:
```
% Context Sampling (Optional)

When `context_sampler` parameter is provided:
1. Measure raw devunit prompt chars BEFORE preprocessing
2. Call `preprocess_with_metadata()` instead of `preprocess()` to capture directive metadata
3. After LLM response, record metrics via sampler:
   - input_chars, output_chars, duration_ms
   - prompt_tokens_reported, response_tokens_reported (from llm_invoke return)
   - preprocessor_items from metadata
4. Works for BOTH cloud and local execution paths

% Function Signature Addition
Add optional parameter:
    context_sampler: Optional[ContextSampler] = None
```

**Behavioral change**: When sampler provided, captures comprehensive metrics without changing generation behavior.

---

### 7. `code_generator_python.prompt` (MODIFY)

**Goal**: Pass through extended return fields from `llm_invoke`.

**Minimal change to "Outputs" section**:
```
% Outputs:
    'runnable_code' - A string that is runnable code
    'total_cost' - A float that is the total cost
    'model_name' - A string that is the model name
    # Pass through from llm_invoke (when available):
    'input_tokens', 'output_tokens', 'duration_ms', 'provider'
```

**Note**: These fields flow through from `llm_invoke()` return dict. No new logic needed.

---

### 8. `sync_orchestration_python.prompt` (MODIFY)

**Goal**: Initialize sampler at workflow start, finalize after generation.

**Add to "Dependencies" section**:
```
<context_sampler_example>
    <include>context/context_sampler_example.py</include>
</context_sampler_example>
```

**Add new section**:
```
% Context Sampling Integration

When context sampling is enabled (via config):
1. At workflow start: Initialize ContextSampler with output path
2. Generate unique generation_id (UUID)
3. Pass sampler to code_generator_main()
4. After generation: Call sampler.finalize() to write context file
5. Log context file path (unless quiet mode)

Configuration:
- context_sampling: bool (default true) - enable/disable
- context_samples: int (default 5) - max files to retain
```

---

## Prompt Dependency Graph

```
context_map.schema.json (canonical schema)
         │
         ▼
context_map_models_python.prompt (Pydantic models)
         │
    ┌────┴────┐
    ▼         ▼
context_sampler    context_viz
    │
    ▼
preprocess (add metadata mode)
    │
    ▼
code_generator_main (instrumentation)
    │
    ▼
sync_orchestration (wire up sampler)
```

## Generation Order

Run `pdd generate` in this order to respect dependencies:

```bash
# 1. New modules (no existing code to break)
pdd generate context_map_models --language python
pdd generate context_sampler --language python
pdd generate context_viz --language python

# 2. Modified modules (regenerate with new requirements)
pdd generate llm_invoke --language python
pdd generate preprocess --language python
pdd generate code_generator --language python
pdd generate code_generator_main --language python
pdd generate sync_orchestration --language python
```

## Context Files to Create

Each modified prompt needs a context example file for the `<include>` directive:

| File | Purpose |
|------|---------|
| `context/context_sampler_example.py` | Shows ContextSampler usage pattern |
| `context/preprocess_metadata_example.py` | Shows preprocess_with_metadata() usage |

## Testing Strategy

1. **Schema compliance**: `python -c "from pdd.context_map_models import ContextMap; print(ContextMap.generate_sample().to_json())"`
2. **Sampler CLI**: `python -m pdd.context_sampler --sample > test.json && python -m pdd.context_viz test.json`
3. **Integration**: Run `pdd sync` on a test prompt, verify `.pdd_context/` files created
4. **Visualization**: Pipe sample through viz: `python -m pdd.context_sampler --sample | python -m pdd.context_viz`

## Configuration

**In `.pddrc`**:
```yaml
defaults:
  context_sampling: true   # Enable/disable context capture
  context_samples: 5       # Max context files to retain per devunit
```

**Environment override**: `PDD_CONTEXT_SAMPLING=0` disables capture.

## Files Summary

| File | Type | Description |
|------|------|-------------|
| `prompts/context_map_models_python.prompt` | NEW | Pydantic models from schema |
| `prompts/context_sampler_python.prompt` | NEW | Storage layer with CLI |
| `prompts/context_viz_python.prompt` | NEW | Visualization CLI |
| `prompts/llm_invoke_python.prompt` | MODIFY | Add token/duration to return |
| `prompts/preprocess_python.prompt` | MODIFY | Add metadata collection mode |
| `prompts/code_generator_python.prompt` | MODIFY | Pass through extended fields |
| `prompts/code_generator_main_python.prompt` | MODIFY | Main instrumentation point |
| `prompts/sync_orchestration_python.prompt` | MODIFY | Wire up sampler lifecycle |
| `context/context_sampler_example.py` | NEW | Usage example for includes |
| `context_map.schema.json` | EXISTING | Canonical JSON schema |

## Prompt Writing Guidelines

When modifying prompts, follow PDD principles:

1. **Behavioral, not procedural**: Describe WHAT the code should do, not HOW
2. **Testable requirements**: Each requirement should map to a test case
3. **10-30% ratio**: Prompt should be 10-30% of expected code size
4. **Use includes**: Reference dependencies via `<include>` tags
5. **Avoid duplication**: Don't repeat what's in preamble or grounding
6. **Positive constraints**: "Return empty dict on error" not "Don't throw exceptions"

## Rollback Strategy

If regenerated code breaks:
1. Prompts are version-controlled - revert prompt changes
2. Regenerate from previous prompt version
3. Tests catch regressions before merge
