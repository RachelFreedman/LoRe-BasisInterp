#!/usr/bin/env python3
"""Print and validate corrected PRISM pairs before generating embeddings."""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from modeling import MODEL_NAME, PAIR_FORMAT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--num-samples", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(
        "parquet",
        data_files=f"data/prism/{args.split}.parquet",
    )["train"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for index in range(min(args.num_samples, len(dataset))):
        entry = dataset[index]
        info = entry["extra_info"]
        chosen_text = info["chosen_utterance"]
        rejected_text = info["rejected_utterance"]

        assert info["pair_format"] == PAIR_FORMAT
        assert isinstance(chosen_text, str)
        assert isinstance(rejected_text, str)

        prompt = entry["prompt"]
        chosen_conv = prompt + [{"role": "assistant", "content": chosen_text}]
        rejected_conv = prompt + [{"role": "assistant", "content": rejected_text}]

        assert [message["role"] for message in chosen_conv] == [
            message["role"] for message in rejected_conv
        ]
        assert all(isinstance(message["content"], str) for message in chosen_conv)
        assert all(isinstance(message["content"], str) for message in rejected_conv)
        assert chosen_conv[:-1] == rejected_conv[:-1]

        chosen_rendered = tokenizer.apply_chat_template(
            chosen_conv,
            tokenize=False,
        )
        rejected_rendered = tokenizer.apply_chat_template(
            rejected_conv,
            tokenize=False,
        )

        print("=" * 88)
        print(
            f"PAIR {index + 1}: dialog={info['dialog_id']} "
            f"turn={info['turn_nb']} rejected_idx={info['rejected_idx']}"
        )
        print("roles:", [message["role"] for message in chosen_conv])
        print("content types (chosen):", [
            type(message["content"]).__name__ for message in chosen_conv
        ])
        print("content types (rejected):", [
            type(message["content"]).__name__ for message in rejected_conv
        ])
        print("\nCHOSEN TEXT:\n", chosen_text, sep="")
        print("\nREJECTED TEXT:\n", rejected_text, sep="")
        print("\nCHOSEN RENDERED CHAT TEMPLATE:\n", chosen_rendered, sep="")
        print("\nREJECTED RENDERED CHAT TEMPLATE:\n", rejected_rendered, sep="")

    print(
        f"\nValidated {min(args.num_samples, len(dataset))} "
        f"{PAIR_FORMAT} pairs."
    )


if __name__ == "__main__":
    main()
