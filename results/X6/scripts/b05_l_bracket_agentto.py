from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b05 L-bracket", "domain": {"size": [100.0, 100.0, 40.0]}, "volume_fraction": 0.3, "supports": [{"id": "ceiling", "region": {"face": "y+", "center": [20.0, 20.0], "size": [40.0, 40.0]}}], "loads": [{"id": "tip", "region": {"face": "x+", "center": [20.0, 20.0], "size": [10.0, 10.0]}, "force": [0.0, -500.0, 0.0]}], "passive": [{"id": "occupied", "kind": "void", "region": {"min": [40.0, 40.0, 0.0], "max": [100.0, 100.0, 40.0]}}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
