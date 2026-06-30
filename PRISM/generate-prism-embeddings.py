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

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import utils

utils.set_seed(42)

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
    start_idx = 0
    checkpoint_path = output_path + ".checkpoint"
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint {checkpoint_path}")
        embeddings_data = torch.load(checkpoint_path)
        start_idx = len(embeddings_data)
        print(f"Skipping first {start_idx} entries...")

    for i, entry in enumerate(tqdm(dataset, desc="Generating embeddings", initial=start_idx, total=len(dataset))):
        if i < start_idx:
            continue
            
        user_id = entry["extra_info"]["user_id"]
        dialog_id = entry["extra_info"]["dialog_id"]
        prompt = entry["prompt"]

        chosen = [{"content": entry["extra_info"]["chosen_utterance"], "role": "assistant"}]
        rejected = [{"content": entry["extra_info"]["rejected_utterance"], "role": "assistant"}]
        chosen_conv = prompt + chosen
        rejected_conv = prompt + rejected
        
        # Tokenize the current dialog state
        tokenized = tokenizer.apply_chat_template(
            chosen_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        entry["extra_info"]["chosen_conv_embedding"] = embedding

        # Tokenize the current dialog state
        tokenized = tokenizer.apply_chat_template(
            rejected_conv,
            tokenize=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = model(tokenized)
            embedding = output.last_hidden_state[0, -1].cpu()  # [hidden_dim]
            del output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        entry["extra_info"]["rejected_conv_embedding"] = embedding

        embeddings_data.append(entry)
        
        # Save a checkpoint every 100 steps to prevent data loss on timeout
        if len(embeddings_data) % 100 == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(embeddings_data, checkpoint_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_data, output_path)
    
    # Clean up checkpoint when fully complete
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    print(f"✅ Saved embeddings to {output_path}")

    return embeddings_data


if __name__ == "__main__":
    # --- Configuration ---
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

    # --- Load model and tokenizer ---
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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
