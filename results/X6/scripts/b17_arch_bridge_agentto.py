from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b17 arch bridge", "domain": {"size": [300.0, 100.0, 60.0]}, "volume_fraction": 0.15, "supports": [{"id": "abutment_left", "region": {"axis": "z", "at": [0.0, 0.0]}}, {"id": "abutment_right", "region": {"axis": "z", "at": [300.0, 0.0]}}], "loads": [{"id": "deck_load", "region": {"face": "y+"}, "force": [0.0, -6000.0, 0.0]}], "passive": [{"id": "deck", "kind": "solid", "region": {"min": [0.0, 94.0, 0.0], "max": [300.0, 100.0, 60.0]}}], "resolution": {"h": 2.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
