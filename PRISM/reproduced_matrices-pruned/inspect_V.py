"""Look at a saved V basis matrix yourself.

Usage:  python inspect_V.py 10      # inspects PRISM_V_lore_K_10_alpha_10000.0.pt
        python inspect_V.py 50
"""
import sys, torch, torch.nn.functional as F

K = sys.argv[1] if len(sys.argv) > 1 else "10"
V = torch.load(f"PRISM_V_lore_K_{K}_alpha_10000.0.pt", map_location="cpu").float()

# make columns = the basis arrows (shape 4096 x kept)
if V.shape[0] < V.shape[1]:
    V = V.t()
print(f"V shape (rows=embedding dim, cols=bases kept): {tuple(V.shape)}")
kept = V.shape[1]

if kept == 1:
    print("Only 1 basis column kept -> trivially one direction.")
    sys.exit()

# angle between every pair of columns: 1.0 means identical direction
Vn = F.normalize(V, dim=0)
cos = Vn.t() @ Vn
print("\nPairwise cosine between basis columns (1.00 = same direction):")
torch.set_printoptions(precision=3, sci_mode=False)
print(cos)

# singular values: how many *independent* directions actually exist
s = torch.linalg.svdvals(V)
print("\nSingular values (energy per independent direction):")
print([round(x, 3) for x in s.tolist()])
print(f"s2/s1 = {(s[1]/s[0]).item():.4f}   (near 0 means effectively ONE direction)")
