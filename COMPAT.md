# HERETIC Compat Fork — torch 2.5 + Local data_files

This is a compatibility fork of [p-e-w/heretic](https://github.com/p-e-w/heretic)
that works on systems still running torch 2.5 and supports pointing
`good_prompts` / `bad_prompts` at local `.txt` / `.jsonl` files.

**Upstream**: https://github.com/p-e-w/heretic (AGPL-3.0-or-later).
This fork preserves the same license.

## What was broken

Running upstream HERETIC 1.2.0 on an env with torch 2.5.1:

```
AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
  at peft/tuners/tuners_utils.py cast_adapter_dtype
  at bitsandbytes.functional (via similar dtype enum iteration)
```

Root cause: `peft~=0.18` allows peft 0.19.x which was built against torch 2.6's
new FP8 dtype enum; `bitsandbytes~=0.49` has the same dependency. On torch 2.5
both fail at import time.

Also: the config's `data_files` field on `[good_prompts]`/`[bad_prompts]` tables
is silently ignored — `load_prompts` in `utils.py` does not read it. So local
text files cannot be used as prompt sources; you have to publish to HF Hub first.

## What changed

### 1. `pyproject.toml` — dep caps for torch 2.5 compatibility

Loosened pins to allow older, torch-2.5-compatible versions:

| Package | Upstream | Compat fork |
|---|---|---|
| `peft` | `~=0.18` | `>=0.16,<0.19` |
| `bitsandbytes` | `~=0.49` | `>=0.44,<0.47` |
| `transformers` | `~=5.3` | `>=4.40,<6.0` |
| `kernels` | `~=0.12` | *removed* (not strictly required) |

Rationale: peft 0.19 and bnb 0.49 both hit `torch.float8_e8m0fnu` which only
exists on torch 2.6+. Older versions work cleanly on torch 2.5.

### 2. `src/heretic/config.py` — `DatasetSpecification.data_files` field

```python
data_files: str | list[str] | None = Field(
    default=None,
    description="For local builders ('text', 'json', 'csv'): local file path(s) to load.",
)
```

### 3. `src/heretic/utils.py` — honor `data_files` in `load_prompts`

Added an explicit branch that, when `data_files` is set on the spec, calls:

```python
load_dataset(path, data_files=data_files, split=split_str,
             verification_mode=VerificationMode.NO_CHECKS)
```

where `path` is expected to be a builder name (`"text"`, `"json"`, `"csv"`) and
`data_files` is the local file path.

## Usage

Config snippet using local files:

```toml
[good_prompts]
dataset = "text"
data_files = "/path/to/good_prompts.txt"
split = "train"
column = "text"

[bad_prompts]
dataset = "text"
data_files = "/path/to/bad_prompts.txt"
split = "train"
column = "text"
```

Each line of the .txt file is one prompt.

## Install

```bash
pip install -e .[research]
```

(The package is renamed `heretic-llm-compat` in `pyproject.toml` to avoid
collision if you have upstream `heretic-llm` also installed.)

## Upstreaming

If upstream maintainers want to take these patches, the `data_files` support
should merge cleanly — it's a pure additive field. The dep bumps are trickier
since upstream uses `uv` with `exclude-newer = "7 days"` pinning.
