# Context Map Implementation - Change Prompts

These prompts describe the modifications needed to add context map capture to PDD.

## Execution Order

| # | Prompt | Target | Type |
|---|--------|--------|------|
| 0 | `00_context_sampler_python.prompt` | `pdd/context_sampler.py` | NEW FILE |
| 1 | `01_extend_llm_invoke_return.change.prompt` | `pdd/llm_invoke.py` | MODIFY |
| 2 | `02_preprocess_with_metadata.change.prompt` | `pdd/preprocess.py` | MODIFY |
| 3 | `03_code_generator_instrumentation.change.prompt` | `pdd/code_generator.py` | MODIFY |
| 3b | `03b_code_generator_main_instrumentation.change.prompt` | `pdd/code_generator_main.py` | MODIFY |
| 4 | `04_sync_orchestration_sampler.change.prompt` | `pdd/sync_orchestration.py` | MODIFY |

**Note:** `sync_main.py` does NOT need modification.

## Call Flow

```
sync_main.py (no changes)
  └─► sync_orchestration.py ──────────────────┐ Initialize sampler
        └─► code_generator_main.py ◄──────────┤ Preprocessing + instrumentation
              ├─► [cloud path] ───────────────┤ Record cloud metrics
              └─► code_generator.py ──────────┤
                    └─► llm_invoke.py ────────┘ Extended return (tokens, duration)
```

## Dependencies

```
00_context_sampler (new file)
     ↓
01_llm_invoke + 02_preprocess (parallel)
     ↓
03_code_generator + 03b_code_generator_main (parallel, depend on 01, 02)
     ↓
04_sync_orchestration (depends on 00, 03, 03b)
```

## Workflow

Since PDD's core modules don't have existing prompts, the workflow is:

1. **Generate prompts from existing code** using `pdd update`
2. **Apply changes** using `pdd change`
3. **Regenerate code** using `pdd generate`

### Step 0: Create new context_sampler module

```bash
cd ~/pdd

# Copy prompt and schema
cp ~/pdd_context_viz/changes/00_context_sampler_python.prompt prompts/context_sampler_python.prompt
cp ~/pdd_context_viz/context_map.schema.json prompts/context_map.schema.json

# Generate new module
pdd generate context_sampler --language python
```

### Steps 1-4: Modify existing modules

For each module, run from `~/pdd`:

```bash
# Example for llm_invoke (repeat pattern for each module)

# 1. Generate prompt from existing code
pdd update pdd/llm_invoke.py --output prompts/llm_invoke_python.prompt

# 2. Apply change
pdd change \
  ~/pdd_context_viz/changes/01_extend_llm_invoke_return.change.prompt \
  pdd/llm_invoke.py \
  prompts/llm_invoke_python.prompt \
  --output prompts/llm_invoke_python.prompt

# 3. Regenerate code
pdd generate llm_invoke --language python
```

### Full execution script

```bash
cd ~/pdd

# 0. New module
cp ~/pdd_context_viz/changes/00_context_sampler_python.prompt prompts/context_sampler_python.prompt
cp ~/pdd_context_viz/context_map.schema.json prompts/context_map.schema.json
pdd generate context_sampler --language python

# 1. llm_invoke
pdd update pdd/llm_invoke.py --output prompts/llm_invoke_python.prompt
pdd change ~/pdd_context_viz/changes/01_extend_llm_invoke_return.change.prompt \
  pdd/llm_invoke.py prompts/llm_invoke_python.prompt \
  --output prompts/llm_invoke_python.prompt
pdd generate llm_invoke --language python

# 2. preprocess
pdd update pdd/preprocess.py --output prompts/preprocess_python.prompt
pdd change ~/pdd_context_viz/changes/02_preprocess_with_metadata.change.prompt \
  pdd/preprocess.py prompts/preprocess_python.prompt \
  --output prompts/preprocess_python.prompt
pdd generate preprocess --language python

# 3. code_generator
pdd update pdd/code_generator.py --output prompts/code_generator_python.prompt
pdd change ~/pdd_context_viz/changes/03_code_generator_instrumentation.change.prompt \
  pdd/code_generator.py prompts/code_generator_python.prompt \
  --output prompts/code_generator_python.prompt
pdd generate code_generator --language python

# 3b. code_generator_main (where preprocessing and cloud handling happens)
pdd update pdd/code_generator_main.py --output prompts/code_generator_main_python.prompt
pdd change ~/pdd_context_viz/changes/03b_code_generator_main_instrumentation.change.prompt \
  pdd/code_generator_main.py prompts/code_generator_main_python.prompt \
  --output prompts/code_generator_main_python.prompt
pdd generate code_generator_main --language python

# 4. sync_orchestration
pdd update pdd/sync_orchestration.py --output prompts/sync_orchestration_python.prompt
pdd change ~/pdd_context_viz/changes/04_sync_orchestration_sampler.change.prompt \
  pdd/sync_orchestration.py prompts/sync_orchestration_python.prompt \
  --output prompts/sync_orchestration_python.prompt
pdd generate sync_orchestration --language python
```

## Testing

After applying all changes:

```bash
cd ~/pdd
pytest tests/ -k "context"  # Run any context-related tests
pdd sync some_test_module   # Verify .pdd_context/ files are created
```

## Rollback

If issues arise, revert using git:

```bash
cd ~/pdd
git checkout -- pdd/llm_invoke.py pdd/preprocess.py pdd/code_generator.py \
  pdd/code_generator_main.py pdd/sync_orchestration.py
rm pdd/context_sampler.py
rm prompts/*_python.prompt  # Remove generated prompts
```
