from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b35 padeye", "domain": {"size": [100.0, 80.0, 30.0]}, "volume_fraction": 0.3, "supports": [{"id": "weld", "region": {"face": "y-"}}], "loads": [{"id": "sling", "region": {"axis": "z", "center": [50.0, 55.0], "radius": 15.0, "span": [0.0, 30.0]}, "force": [2500.0, 6000.0, 0.0]}], "passive": [{"id": "boss", "kind": "solid", "region": {"axis": "z", "center": [50.0, 55.0], "radius": 15.0, "span": [0.0, 30.0]}}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
