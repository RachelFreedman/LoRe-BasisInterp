"""
Modal wrapper for embed_response_pool.py.

  modal run modal_embed_pool.py --limit 20   # smoke test first
  modal run modal_embed_pool.py

Mounts only the script; the pool and prompt JSON travel as function arguments, so nothing of
the 3.1 GB working tree is uploaded.
"""

import os
import subprocess

import modal

app = modal.App("lore-synth-pool-embed")

prism_volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.5.1", "transformers==4.47.0", "accelerate==1.2.1", "tqdm")
    .add_local_file("PRISM/embed_response_pool.py",
                    remote_path="/workspace/PRISM/embed_response_pool.py")
)


@app.function(
    image=image,
    volumes={"/vol": prism_volume},
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(pool_json: bytes, prompts_json: bytes, limit: int = 0, batch_size: int = 16,
        token_budget: int = 12000) -> bytes:
    os.chdir("/workspace")
    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    os.makedirs("data/synthetic_personas", exist_ok=True)
    with open("data/synthetic_personas/response_pool.json", "wb") as f:
        f.write(pool_json)
    with open("data/synthetic_personas/prompts.json", "wb") as f:
        f.write(prompts_json)

    cmd = ["python", "PRISM/embed_response_pool.py", "--batch_size", str(batch_size),
           "--token_budget", str(token_budget)]
    if limit:
        cmd += ["--limit", str(limit)]
    subprocess.run(cmd, check=True)

    with open("data/synthetic_personas/pool_embeddings.pt", "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 16, token_budget: int = 12000):
    base = "data/synthetic_personas"
    with open(f"{base}/response_pool.json", "rb") as f:
        pool_json = f.read()
    with open(f"{base}/prompts.json", "rb") as f:
        prompts_json = f.read()

    out_name = "pool_embeddings_smoke.pt" if limit else "pool_embeddings.pt"
    print(f"submitting to Modal (limit={limit or 'all'}) -> {base}/{out_name}")
    blob = run.remote(pool_json, prompts_json, limit=limit, batch_size=batch_size,
                      token_budget=token_budget)

    os.makedirs(base, exist_ok=True)
    with open(f"{base}/{out_name}", "wb") as f:
        f.write(blob)
    print(f"wrote {base}/{out_name} ({len(blob)} bytes)")
