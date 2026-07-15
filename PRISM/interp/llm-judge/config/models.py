"""Judge model registry: short alias -> provider + concrete model id.

`--models <alias> ...` on the CLI selects from these. The default run uses all
three families (one round each). Each model_id must be a concrete string; any alias
left at UNSET_MODEL_ID is rejected before the run makes a single API call, so an
unfinished config fails fast instead of burning quota on a nonexistent model.
"""

# Sentinel meaning "no model id has been chosen for this alias yet".
UNSET_MODEL_ID = "<unset>"

JUDGE_MODELS: dict[str, dict[str, str]] = {
    "claude": {"provider": "anthropic", "model_id": "claude-opus-4-8"},
    "gemini": {"provider": "google", "model_id": "gemini-3.5-flash"},
    "chatgpt": {"provider": "openai", "model_id": "gpt-5.6-sol"},
}

# Default set of aliases when --models is not passed: one round per LLM family.
DEFAULT_MODELS: list[str] = ["claude", "gemini", "chatgpt"]
