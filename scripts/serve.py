"""Start the agentto backend (127.0.0.1:8765) without installing the package.
    python -B scripts/serve.py [--ladder-max 1.2e7]     (a lower ladder cap, for GUI tests on small meshes)"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":                     # the GPU worker is a spawned process: guard the entry point
    if "--ladder-max" in sys.argv:
        i = sys.argv.index("--ladder-max")
        os.environ["AGENTTO_LADDER_MAX"] = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    from agentto.app.server import main
    main()
