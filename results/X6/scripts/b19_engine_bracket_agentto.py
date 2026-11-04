from agentto.tools import Session
s = Session(None)
s.call('set_spec', {'spec': {"name": "b19 engine bracket", "units": {"length": "in", "force": "lbf"}, "domain": {"size": [7.0, 2.5, 4.0]}, "material": {"name": "Ti-6Al-4V", "E": 16500000.0, "nu": 0.34}, "volume_fraction": 0.2, "supports": [{"id": "bolt_1", "region": {"face": "y-", "center": [0.5, 0.5], "size": [0.5, 0.5], "shape": "ellipse"}}, {"id": "bolt_2", "region": {"face": "y-", "center": [0.5, 3.5], "size": [0.5, 0.5], "shape": "ellipse"}}, {"id": "bolt_3", "region": {"face": "y-", "center": [2.5, 0.5], "size": [0.5, 0.5], "shape": "ellipse"}}, {"id": "bolt_4", "region": {"face": "y-", "center": [2.5, 3.5], "size": [0.5, 0.5], "shape": "ellipse"}}], "loads": [{"id": "pull", "region": {"axis": "z", "center": [6.0, 1.75], "radius": 0.5, "span": [1.25, 2.75]}, "force": [0.0, 8000.0, 0.0]}], "passive": [{"id": "clevis", "kind": "solid", "region": {"axis": "z", "center": [6.0, 1.75], "radius": 0.5, "span": [1.25, 2.75]}}], "resolution": {"h": 0.05}}})
job = s.call('solve', {'rung': 'release', 'optimize': True})['job']
s.call('wait_job', {'job': job})
s.call('export_stl', {'job': job})
