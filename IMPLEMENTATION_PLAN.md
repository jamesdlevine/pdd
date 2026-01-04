# Context Map Implementation Plan for PDD

## Overview

Add context map capture to `pdd generate`/`pdd sync` to log detailed metrics about each generation for analysis and visualization.

## Key Files in PDD

| File | Role |
|------|------|
| `pdd/preprocess.py` | Expands `<include>`, `<shell>`, `<web>`, backticks, variables |
| `pdd/code_generator_main.py` | CLI wrapper: handles cloud vs local, preprocessing, incremental |
| `pdd/code_generator.py` | Core local generation: calls llm_invoke |
| `pdd/llm_invoke.py` | Calls LiteLLM, captures tokens via callback in `_LAST_CALLBACK_DATA` |
| `pdd/sync_main.py` | Entry point for `pdd sync`, iterates over languages (NO CHANGES) |
| `pdd/sync_orchestration.py` | Per-language sync workflow, calls code_generator_main |

## Actual Data Flow

```
sync_main.py (NO CHANGES - just iterates languages)
    → sync_orchestration.py (initialize sampler, finalize after)
        → code_generator_main.py (preprocessing happens HERE, cloud path HERE)
            ├─► [cloud path] → HTTP request (bypass code_generator.py)
            └─► code_generator.py (local path)
                    → llm_invoke() → {'result', 'cost', 'model_name', ...}
```

**Critical insight**: `code_generator_main.py` does preprocessing BEFORE calling `code_generator.py`, and cloud execution bypasses `code_generator.py` entirely.

## Implementation Steps

### 1. Create `pdd/context_sampler.py` (NEW)

Storage layer implementing the schema. Key interface:

```python
class ContextSampler:
    def __init__(self, output_path: str, max_samples: int = 5): ...
    def start_generation(self, generation_id: str, prompt_file: str, model: str, provider: str): ...
    def record_call(self, input_chars, output_chars, duration_ms, tokens...): ...
    def record_prompt_breakdown(self, pdd_system_chars, devunit_chars, ...): ...
    def finalize(self) -> str: ...
```

### 2. Extend `pdd/preprocess.py`

**Goal**: Return preprocessing metadata alongside expanded text.

**Changes**:
- Add `PreprocessMetadata` dataclass to track items
- Modify `process_include_tags()`, `process_shell_tags()`, `process_web_tags()`, etc. to collect metadata
- Add new function `preprocess_with_metadata()` that returns both text and metadata
- Keep existing `preprocess()` unchanged for backward compatibility

```python
@dataclass
class PreprocessorItem:
    type: str  # "include", "shell", "web", "variable"
    source: str  # file path, command, url, or var name
    chars: int
    line_in_prompt: int
    syntax: Optional[str] = None  # "directive" or "backticks" for includes
    include_many: bool = False

@dataclass
class PreprocessMetadata:
    items: List[PreprocessorItem]
    summary: Dict[str, int]  # {include_count, include_chars, shell_count, ...}

def preprocess_with_metadata(prompt: str, ...) -> Tuple[str, PreprocessMetadata]:
    # New function that tracks all expansions
    ...
```

### 3. Extend `pdd/llm_invoke.py` return value

**Goal**: Expose token counts and duration in return dict.

**Changes** (minimal):
- Add timing around litellm call (already has `start_time`/`end_time` locally)
- Extend return dict:

```python
return {
    'result': final_result,
    'cost': total_cost,
    'model_name': model_name_litellm,
    'thinking_output': final_thinking,
    # NEW:
    'input_tokens': _LAST_CALLBACK_DATA.get("input_tokens", 0),
    'output_tokens': _LAST_CALLBACK_DATA.get("output_tokens", 0),
    'duration_ms': int((end_time - start_time) * 1000),
    'provider': provider_name,  # extract from model_info
}
```

### 4. Modify `pdd/code_generator_main.py` (CRITICAL)

**Goal**: Instrument preprocessing and record metrics for BOTH cloud and local paths.

This is where preprocessing actually happens. The function:
1. Reads and preprocesses the prompt
2. Either sends to cloud OR calls `code_generator.py`
3. Handles the response

**Changes**:

```python
def code_generator_main(ctx, prompt_file, output, ..., context_sampler=None):
    # Measure raw devunit prompt BEFORE preprocessing
    devunit_chars = len(prompt_content)

    # Use preprocess_with_metadata when sampler provided
    if context_sampler:
        processed, preprocess_meta = preprocess_with_metadata(prompt_content, ...)
    else:
        processed = pdd_preprocess(prompt_content, ...)
        preprocess_meta = None

    # ... cloud or local execution ...

    # Record metrics (works for both cloud and local paths)
    if context_sampler:
        context_sampler.record_call(
            input_chars=len(processed),
            output_chars=len(generated_code_content),
            duration_ms=response.get('duration_ms', 0),
            prompt_tokens_reported=response.get('input_tokens'),
            response_tokens_reported=response.get('output_tokens'),
        )
        if preprocess_meta:
            context_sampler.record_prompt_breakdown(
                devunit_prompt_chars=devunit_chars,
                preprocessor_items=preprocess_meta.items,
            )
```

### 5. Modify `pdd/code_generator.py` (minimal)

**Goal**: Pass through extended return from llm_invoke.

Since preprocessing happens in `code_generator_main.py`, this function just needs to:
1. Accept optional `context_sampler` parameter
2. Pass through extended fields from `llm_invoke()` return

```python
def code_generator(prompt, language, strength, ..., context_sampler=None):
    # ... existing code ...
    response = llm_invoke(...)

    # Optionally record if sampler provided (for direct calls not via code_generator_main)
    if context_sampler:
        context_sampler.record_call(...)

    return response['result'], response['cost'], response['model_name']
```

### 6. Wire up in `pdd/sync_orchestration.py`

**Goal**: Initialize sampler and finalize after generation.

**Changes**:

```python
from .context_sampler import ContextSampler

def sync_orchestration(...):
    # Initialize sampler
    sampler = ContextSampler(
        output_path=code_output_path,
        max_samples=config.get('context_samples', 5)
    )
    generation_id = str(uuid.uuid4())
    sampler.start_generation(generation_id, prompt_file, model, provider)

    # Call code_generator_main with sampler
    result = code_generator_main(
        ctx=ctx,
        prompt_file=str(pdd_files['prompt'].resolve()),
        output=str(pdd_files['code'].resolve()),
        ...,
        context_sampler=sampler,
    )

    # Finalize
    context_path = sampler.finalize()
    if not quiet:
        console.print(f"[dim]Context saved: {context_path}[/dim]")
```

### 7. Add config option

**In `.pddrc`**:
```yaml
defaults:
  context_sampling: true   # Enable/disable
  context_samples: 5       # Max samples to retain
```

## File Changes Summary

| File | Type | Description |
|------|------|-------------|
| `pdd/context_sampler.py` | NEW | Storage layer implementation |
| `pdd/preprocess.py` | MODIFY | Add `preprocess_with_metadata()`, track items |
| `pdd/llm_invoke.py` | MODIFY | Add tokens/duration to return dict (~5 lines) |
| `pdd/code_generator_main.py` | MODIFY | Main instrumentation: preprocessing metadata, metrics |
| `pdd/code_generator.py` | MODIFY | Pass through extended llm_invoke return |
| `pdd/sync_orchestration.py` | MODIFY | Initialize/finalize sampler |
| `pdd/sync_main.py` | NO CHANGE | Just iterates languages |
| `.pddrc` schema | MODIFY | Add context_sampling options |

## Data Capture Points

| Data | Location | How |
|------|----------|-----|
| `generation_id` | sync_orchestration | Generate UUID |
| `timestamp_utc` | context_sampler.start_generation | `datetime.utcnow()` |
| `duration_ms` | llm_invoke | `end_time - start_time` |
| `model`, `provider` | llm_invoke return | Already available |
| `prompt_file` | sync_orchestration | Passed from CLI |
| `devunit_prompt_chars` | code_generator_main | `len(prompt)` before preprocess |
| `preprocessor_items` | preprocess_with_metadata | Track each expansion |
| `input_tokens`, `output_tokens` | llm_invoke | From `_LAST_CALLBACK_DATA` |
| `response_chars` | code_generator_main | `len(generated_code_content)` |

## Testing Strategy

1. Unit tests for `context_sampler.py` (file operations, retention)
2. Unit tests for `preprocess_with_metadata()` (metadata accuracy)
3. Integration test: run `pdd sync` and verify `.pdd_context/` files created
4. Verify JSON validates against schema

## Optional Enhancements

1. **Environment variable toggle**: `PDD_CONTEXT_SAMPLING=0` to disable
2. **Verbose output**: Show context summary after generation
3. **Sampling rate**: For high-volume, sample 1-in-N generations
