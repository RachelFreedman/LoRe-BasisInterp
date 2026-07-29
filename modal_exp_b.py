"""
Modal runner for Experiment B (causal edit probe).

Applies single-attribute LLM edits to rejected responses (via AWS Bedrock), re-embeds them through
Skywork on an A10G GPU, and measures the score change along the PRISM direction, then syncs the CSV.

Prereqs (two Modal secrets):
  * "huggingface" : HF_TOKEN (for Skywork + the PRISM dataset)      [mirrors modal_compute_vectors.py]
  * "aws"         : AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY with Bedrock access.
                    generate_contrastive_pairs.py pins region us-east-1; override with AWS creds that
                    can call Claude Sonnet in that region (or edit the region in that file).
Run:      modal run modal_exp_b.py            # default 60 rejected responses x 5 edits
          modal run modal_exp_b.py --n 120
"""
import os
import subprocess

import modal

app = modal.App("lore-prism-exp-b-causal-edit")
prism_volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.0",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "boto3",            # Bedrock client for the LLM edits
        "tqdm",
    )
    .add_local_dir(".", remote_path="/workspace")
)


@app.function(          # CPU only: one Bedrock call to validate creds/region/model, no GPU
    image=image,
    timeout=600,
    secrets=[modal.Secret.from_name("aws")],
)
def dry_run_exp_b():
    os.chdir("/workspace")
    subprocess.run(["python", "PRISM/exp_b_causal_edit.py", "--dry-run"], check=True)


@app.function(
    image=image,
    volumes={"/vol": prism_volume},
    gpu="A10G",
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("aws")],
)
def run_exp_b(n: int):
    os.chdir("/workspace")
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    subprocess.run(["python", "PRISM/exp_b_causal_edit.py", "--n", str(n)], check=True)

    out = "/workspace/results/causal_edit/exp_b_causal_edit.csv"
    with open(out, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(n: int = 60, dry_run: bool = False):
    if dry_run:
        print("Validating Bedrock creds/region/model on Modal (CPU, one call)...")
        dry_run_exp_b.remote()
        return
    print("Submitting Experiment B (causal edit probe) to Modal...")
    csv_bytes = run_exp_b.remote(n)
    os.makedirs("results/causal_edit", exist_ok=True)
    with open("results/causal_edit/exp_b_causal_edit.csv", "wb") as f:
        f.write(csv_bytes)
    print("Synced results/causal_edit/exp_b_causal_edit.csv back to your machine.")
