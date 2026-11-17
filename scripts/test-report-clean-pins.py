#!/usr/bin/env python3
"""Ensure timing reports reject contaminated, failed and incomplete runs."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('report',root/'scripts/report-clean-pins.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class GateTests(unittest.TestCase):
 def test_contaminated_exit_zero_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'ai-run.json').write_text(json.dumps(dict(verdict='CONTAMINATED',exit=0)));sys.argv=['report',tmp]
   with self.assertRaises(AssertionError):m.main()
 def test_clean_failure_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'ai-run.json').write_text(json.dumps(dict(verdict='CLEAN',exit=101)));sys.argv=['report',tmp]
   with self.assertRaises(AssertionError):m.main()
 def test_incomplete_correctness_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'ai-run.json').write_text(json.dumps(dict(verdict='CLEAN',exit=0)));(p/'validation.json').write_text(json.dumps(dict(complete=False)));sys.argv=['report',tmp]
   with self.assertRaises(AssertionError):m.main()
unittest.main()
