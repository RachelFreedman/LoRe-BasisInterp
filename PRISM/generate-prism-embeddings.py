#!/usr/bin/env python3
#
# Source: https://github.com/facebookresearch/LoRe/blob/main/PRISM/prepare.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import torch
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from modeling import MODEL_NAME, PAIR_FORMAT

# Named function to replace lambda for pickle compatibility
def nested_defaultdict():
    return defaultdict(list)

# Optional: convert nested defaultdicts to regular dicts for clean pickling
def recursive_dict(d):
    if isinstance(d, defaultdict):
        return {k: recursive_dict(v) for k, v in d.items()}
    return d

def generate_prism_embeddings(
    dataset,
    model,
    tokenizer,
    device,
    output_path
):
    """
    Generate embeddings for each user in the dataset.
    Structure: chosen_embeddings[user_id][dialog_id] = [embedding_turn_0, ..., embedding_turn_n]

    Alternate:

    embeddings[user_id][dialog_id][turn_nb][chosen/rejected][seen : True or False][train : True or False]

    Later for given user_id (and specifiec chosen/rejected value, seen True or False value) gather all chosen embeddings as a tensor
    """
    embeddings_data = []
    chosen_embedding_cache = {}
    for entry in tqdm(dataset, desc="Generating embeddings"):
        
        user_id = entry["extra_info"]["user_id"]
        dialog_id = entry["extra_info"]["dialog_id"]
        prompt = entry["prompt"]
        chosen_text = entry["extra_info"]["chosen_utterance"]
        rejected_text = entry["extra_info"]["rejected_utterance"]

        if entry["extra_info"].get("pair_format") != PAIR_FORMAT:
            raise ValueError(
                "Dataset does not use the corrected string-vs-string PRISM "
                "pair format. Rerun PRISM/prepare.py before embedding."
            )
        if not isinstance(chosen_text, str) or not isinstance(rejected_text, str):
            raise TypeError(
                "PRISM chosen_utterance and rejected_utterance must both be "
                f"strings; got {type(chosen_text).__name__} and "
                f"{type(rejected_text).__name__}."
            )

        chosen = [{"content": chosen_text, "role": "assistant"}]
        rejected = [{"content": rejected_text, "role": "assistant"}]
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected

        # A turn with multiple rejected responses produces multiple rows but
        # shares one chosen conversation, so only embed that side once.
        chosen_key = (user_id, dialog_id, entry["extra_info"]["turn_nb"])
        if chosen_key not in chosen_embedding_cache:
            tokenized = tokenizer.apply_chat_template(
                chosen_conv,
                tokenize=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model(tokenized)
                chosen_embedding_cache[chosen_key] = (
                    output.last_hidden_state[0, -1].cpu()
                )

        entry["extra_info"]["chosen_conv_embedding"] = (
            chosen_embedding_cache[chosen_key]
        )

        # Tokenize the current dialog state
        tokenized = tokenizer.apply_chat_template(
            rejected_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]

        entry["extra_info"]["rejected_conv_embedding"] = embedding

        embeddings_data.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_data, output_path)
    print(f"✅ Saved embeddings to {output_path}")

    return embeddings_data


if __name__ == "__main__":
    # --- Configuration ---
    device = "cuda:0"
    model_name = MODEL_NAME

    # --- Load model and tokenizer ---
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Load datasets ---
    print("📦 Loading datasets...")
    train_dataset = load_dataset("parquet", data_files="data/prism/train.parquet")["train"]
    test_dataset = load_dataset("parquet", data_files="data/prism/test.parquet")["train"]

    # # --- Generate embeddings ---
    train_embeddings = generate_prism_embeddings(train_dataset, model, tokenizer, device, "data/prism/train_embeddings.pkl")
    test_embeddings = generate_prism_embeddings(test_dataset, model, tokenizer, device, "data/prism/test_embeddings.pkl")
