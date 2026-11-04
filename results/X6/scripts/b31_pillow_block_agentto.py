from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b31 pillow block", "domain": {"size": [120.0, 60.0, 40.0]}, "material": {"name": "grey cast iron", "E": 110000.0, "nu": 0.26}, "volume_fraction": 0.3, "supports": [{"id": "bolt_left", "region": {"face": "y-", "center": [15.0, 20.0], "size": [16.0, 16.0], "shape": "ellipse"}}, {"id": "bolt_right", "region": {"face": "y-", "center": [105.0, 20.0], "size": [16.0, 16.0], "shape": "ellipse"}}], "loads": [{"id": "shaft", "region": {"axis": "z", "center": [60.0, 40.0], "radius": 15.0, "span": [0.0, 40.0]}, "force": [0.0, -3000.0, 0.0]}], "passive": [{"id": "seat", "kind": "solid", "region": {"axis": "z", "center": [60.0, 40.0], "radius": 15.0, "span": [0.0, 40.0]}}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
