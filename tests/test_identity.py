"""X0 identity (host part): specs of the FEAD problems give bit-identical voxfem problems."""
import numpy as np
import pytest

from agentto import library
from agentto.problem import build
from agentto._voxfem import voxfem  # noqa: F401
from voxfem.core import PROBLEMS


@pytest.mark.parametrize("name", ["cantilever", "bridge", "torsion"])
@pytest.mark.parametrize("k", [8, 10, 16, 20, 40, 80, 96])
def test_fead_problem_identity(name, k):
    ref = PROBLEMS[name](k)
    sp = build(library.fead(name, k))
    got = sp.problem
    assert got.grid == ref.grid
    assert np.array_equal(got.fixed, ref.fixed)
    assert np.array_equal(got.load.idx, ref.load.idx)
    assert np.array_equal(got.load.val, ref.load.val)      # exact, not approximate
    assert sp.solid is None and sp.void is None
    assert sp.notes == []
