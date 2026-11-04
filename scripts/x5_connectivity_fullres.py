"""X5: connectivity of the binarized span-bridge designs at full resolution (the fields rendered in Fig. S2).

results/X5/resolution.json reports Betti numbers on a 2x-pooled mask (betti_pool 2). This script labels the
rho >= 0.5 solid and the complementary void of the saved 32.768M-element fields directly, with 6- and 26-connectivity,
and counts void regions that do not touch the domain boundary (enclosed cavities).

Output: results/X5/connectivity_fullres.json
"""
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "results/X5/fields"
FULL = np.ones((3, 3, 3), bool)


def boundary_labels(lab):
    faces = [lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]]
    return set(np.unique(np.concatenate([f.ravel() for f in faces]))) - {0}


out = {}
for arm in ("cold", "reopen_long"):
    rho = np.load(F / f"span_bridge_{arm}_h1.npy")
    solid = rho >= 0.5
    _, s6 = nd.label(solid)
    _, s26 = nd.label(solid, structure=FULL)
    vlab, nv = nd.label(~solid)          # 6-connected void, the dual of 26-connected solid
    enclosed = nv - len(boundary_labels(vlab))
    out[arm] = {"shape": list(rho.shape), "solid_components_6": int(s6), "solid_components_26": int(s26),
                "void_regions_6": int(nv), "enclosed_cavities_6": int(enclosed)}
    print(arm, out[arm])
dst = ROOT / "results/X5/connectivity_fullres.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("->", dst)
