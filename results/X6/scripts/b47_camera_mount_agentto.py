from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b47 camera mount", "domain": {"size": [120.0, 60.0, 60.0]}, "material": {"name": "aluminium 6061", "E": 69000.0, "nu": 0.33}, "volume_fraction": 0.15, "supports": [{"id": "gantry", "region": {"face": "x-"}}], "loads": [{"id": "camera", "region": {"face": "x+", "center": [30.0, 30.0], "size": [20.0, 20.0]}, "force": [0.0, -80.0, 0.0]}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
