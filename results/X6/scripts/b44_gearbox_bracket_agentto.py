from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b44 gearbox bracket", "domain": {"size": [160.0, 80.0, 80.0]}, "volume_fraction": 0.3, "supports": [{"id": "gearbox", "region": {"face": "x-"}}], "loads": [{"id": "lug_top", "region": {"face": "x+", "center": [72.0, 40.0], "size": [16.0, 16.0]}, "force": [0.0, 0.0, 625.0]}, {"id": "lug_bottom", "region": {"face": "x+", "center": [8.0, 40.0], "size": [16.0, 16.0]}, "force": [0.0, 0.0, -625.0]}], "resolution": {"h": 2.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
