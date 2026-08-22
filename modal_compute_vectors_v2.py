"""
Modal wrapper for compute_concept_vectors_v2.py.

Separate from modal_compute_vectors.py because that one hardcodes concept_vectors.pt as its
output, and the existing vectors back committed results.

Uploads contrastive_pairs_v2.json from the local machine rather than expecting it on the volume,
since it is generated locally via Bedrock.

  modal run modal_compute_vectors_v2.py --limit 3   # smoke test first
  modal run modal_compute_vectors_v2.py
"""

import os
import subprocess

import modal

app = modal.App("lore-prism-concept-vectors-v2")

prism_volume = modal.Volume.from_name("lore-prism-data", create_if_missing=True)

# Mount only the compute script. add_local_dir(".") would upload the whole 3.1 GB working tree
# (data/ is 1.3 GB, new_embeddings/ 620 MB) for a script with no local imports. The pairs JSON
# travels as a function argument instead.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.0",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "tqdm",
    )
    .add_local_file("PRISM/compute_concept_vectors_v2.py",
                    remote_path="/workspace/PRISM/compute_concept_vectors_v2.py")
)


@app.function(
    image=image,
    volumes={"/vol": prism_volume},
    gpu="A10G",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run(pairs_json: bytes, limit: int = 0, batch_size: int = 8,
        want_embeddings: bool = False) -> bytes:
    os.chdir("/workspace")

    os.makedirs("/root/.cache", exist_ok=True)
    if not os.path.exists("/root/.cache/huggingface"):
        os.symlink("/vol/huggingface_cache", "/root/.cache/huggingface")

    # Write the uploaded pairs into the container rather than the volume; the volume copy is not
    # needed and keeping it local avoids a write to shared state.
    os.makedirs("data/prism", exist_ok=True)
    with open("data/prism/contrastive_pairs_v2.json", "wb") as f:
        f.write(pairs_json)

    cmd = ["python", "PRISM/compute_concept_vectors_v2.py",
           "--batch_size", str(batch_size)]
    if limit:
        cmd += ["--limit", str(limit)]
    if want_embeddings:
        cmd += ["--save_embeddings", "data/prism/pair_embeddings.pt"]
    subprocess.run(cmd, check=True)

    target = ("data/prism/pair_embeddings.pt" if want_embeddings
              else "data/prism/concept_vectors_v2.pt")
    with open(target, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = 8, pairs: str = "", out: str = "",
         embeddings: bool = False):
    local_pairs = pairs or "data/prism/contrastive_pairs_v2.json"
    if not os.path.exists(local_pairs):
        raise SystemExit(f"{local_pairs} not found -- run generate_contrastive_pairs_v2.py first")
    with open(local_pairs, "rb") as f:
        pairs_json = f.read()

    out_name = out or ("pair_embeddings_v2.pt" if embeddings else
                       ("concept_vectors_v2_smoke.pt" if limit else "concept_vectors_v2.pt"))
    print(f"submitting to Modal (limit={limit or 'all'}, src={local_pairs}) "
          f"-> data/prism/{out_name}")

    blob = run.remote(pairs_json, limit=limit, batch_size=batch_size,
                      want_embeddings=embeddings)

    os.makedirs("data/prism", exist_ok=True)
    with open(f"data/prism/{out_name}", "wb") as f:
        f.write(blob)
    print(f"wrote data/prism/{out_name} ({len(blob)} bytes)")
