"""Shared model helpers for the PRISM LoRe pipeline."""

import torch


MODEL_NAME = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
PAIR_FORMAT = "string-vs-string-v1"


def extract_reward_direction(model, hidden_size=4096):
    """Return the scalar reward head as a [hidden_size, 1] float tensor.

    Skywork is loaded as a sequence-classification model, whose scalar head is
    normally exposed as ``score``. Named fallbacks cover equivalent Hugging
    Face reward-model conventions without silently selecting an internal
    transformer projection.
    """
    candidates = []
    for name in ("score", "reward_head", "v_head", "classifier"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Linear):
            candidates.append((name, module))

    if not candidates:
        for name, module in model.named_modules():
            leaf_name = name.rsplit(".", 1)[-1]
            if (
                leaf_name in {"score", "reward_head", "v_head", "classifier"}
                and isinstance(module, torch.nn.Linear)
            ):
                candidates.append((name, module))

    valid = [
        (name, module)
        for name, module in candidates
        if tuple(module.weight.shape) == (1, hidden_size)
    ]
    if len(valid) != 1:
        shapes = [(name, tuple(module.weight.shape)) for name, module in candidates]
        raise RuntimeError(
            "Expected exactly one scalar reward head with weight shape "
            f"(1, {hidden_size}); found valid={len(valid)}, candidates={shapes}. "
            "Refusing to use an arbitrary internal Linear layer."
        )

    name, head = valid[0]
    direction = head.weight.detach().to(dtype=torch.float32).T.contiguous()
    if tuple(direction.shape) != (hidden_size, 1):
        raise RuntimeError(
            f"Reward head {name!r} produced shape {tuple(direction.shape)}, "
            f"expected ({hidden_size}, 1)."
        )
    print(f"Using scalar reward head {name!r}: {tuple(head.weight.shape)} -> "
          f"V_sft {tuple(direction.shape)}")
    return direction


def load_reference_direction(device, model_name=MODEL_NAME, hidden_size=4096):
    """Load the pretrained sequence-classification head used as LoRe's anchor."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    return extract_reward_direction(model, hidden_size=hidden_size).to(device)
