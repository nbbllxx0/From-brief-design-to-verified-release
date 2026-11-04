from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b11 cantilever line load", "domain": {"size": [120.0, 40.0, 20.0]}, "volume_fraction": 0.3, "supports": [{"id": "wall", "region": {"face": "x-"}}], "loads": [{"id": "tip_edge", "region": {"axis": "z", "at": [120.0, 0.0]}, "force": [0.0, -1000.0, 0.0]}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
