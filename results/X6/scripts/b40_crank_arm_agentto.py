from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b40 crank arm", "domain": {"size": [180.0, 40.0, 30.0]}, "material": {"name": "aluminium 7075", "E": 71700.0, "nu": 0.33}, "volume_fraction": 0.3, "supports": [{"id": "spindle", "region": {"axis": "z", "center": [20.0, 20.0], "radius": 10.0, "span": [0.0, 30.0]}}], "loads": [{"id": "pedal", "region": {"face": "z+", "center": [160.0, 20.0], "size": [16.0, 16.0], "shape": "ellipse"}, "force": [0.0, -1200.0, 0.0]}], "passive": [{"id": "pedal_boss", "kind": "solid", "region": {"axis": "z", "center": [160.0, 20.0], "radius": 8.0, "span": [0.0, 30.0]}}], "resolution": {"h": 1.0}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
