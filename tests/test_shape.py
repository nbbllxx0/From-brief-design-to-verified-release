"""Shape metrics used by X1: Betti numbers, IoU, Hausdorff, pooling."""
import numpy as np

from agentto.shape import betti, compare, pool, pool_factor


def test_betti_of_simple_solids():
    cube = np.zeros((8, 8, 8), bool)
    cube[2:6, 2:6, 2:6] = True
    assert betti(cube) == {"b0": 1, "b1": 0, "b2": 0, "chi": 1}
    ring = cube.copy()
    ring[3:5, 3:5, :] = False                              # square hole through z: one tunnel
    assert betti(ring)["b1"] == 1 and betti(ring)["b0"] == 1
    shell = np.zeros((8, 8, 8), bool)
    shell[1:7, 1:7, 1:7] = True
    shell[3:5, 3:5, 3:5] = False                           # enclosed cavity
    assert betti(shell) == {"b0": 1, "b1": 0, "b2": 1, "chi": 2}
    two = np.zeros((8, 8, 8), bool)
    two[0:2, 0:2, 0:2] = two[2:4, 2:4, 2:4] = True         # corner contact counts as connected
    assert betti(two)["b0"] == 1


def test_compare_and_pool():
    a = np.zeros((16, 16, 16), bool)
    a[4:12, 4:12, 4:12] = True
    b = np.roll(a, 2, axis=0)
    r = compare(a, b)
    assert abs(r["iou"] - 6 / 10) < 1e-12 and r["hausdorff"]["max"] == 2.0
    assert compare(a, a)["iou"] == 1.0 and compare(a, a)["hausdorff"]["max"] == 0.0
    assert pool(a.astype(np.float32), 2).shape == (8, 8, 8)
    assert pool_factor((640, 320, 320)) == 2 and pool_factor((160, 80, 80)) == 1
