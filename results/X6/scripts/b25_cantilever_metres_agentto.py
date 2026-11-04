from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b25 cantilever metres", "units": {"length": "m", "force": "kN"}, "domain": {"size": [2.0, 1.0, 1.0]}, "material": {"name": "structural steel", "E": 210000000.0}, "volume_fraction": 0.15, "supports": [{"id": "wall", "region": {"face": "x-"}}], "loads": [{"id": "bearing", "region": {"face": "x+", "center": [0.5, 0.5], "size": [0.2, 0.2]}, "force": [0.0, -100.0, 0.0]}], "resolution": {"h": 0.02}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
