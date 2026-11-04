from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b02 MBB half", "domain": {"size": [120.0, 40.0, 40.0]}, "material": {"name": "aluminium 6061", "E": 69000.0, "nu": 0.33}, "volume_fraction": 0.3, "supports": [{"id": "end_support", "region": {"axis": "z", "at": [0.0, 0.0]}, "fix": ["y", "z"]}, {"id": "symmetry", "region": {"face": "x+"}, "fix": ["x"]}], "loads": [{"id": "midspan", "region": {"axis": "z", "at": [120.0, 40.0]}, "force": [0.0, -1000.0, 0.0]}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
