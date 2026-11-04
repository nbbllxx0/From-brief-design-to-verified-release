from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b49 hinged arm", "domain": {"size": [200.0, 40.0, 20.0]}, "volume_fraction": 0.3, "supports": [{"id": "hinge", "region": {"axis": "z", "at": [10.0, 20.0]}}], "loads": [{"id": "tip", "region": {"face": "x+", "center": [20.0, 10.0], "size": [10.0, 20.0]}, "force": [0.0, -400.0, 0.0]}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
