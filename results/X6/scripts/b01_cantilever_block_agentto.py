from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b01 cantilever block", "domain": {"size": [200.0, 100.0, 100.0]}, "volume_fraction": 0.3, "supports": [{"id": "wall", "region": {"face": "x-"}}], "loads": [{"id": "pad", "region": {"face": "x+", "center": [50.0, 50.0], "size": [10.0, 10.0]}, "force": [0.0, -1000.0, 0.0]}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
