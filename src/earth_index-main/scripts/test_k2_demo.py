"""Independent synthetic checks of the pinned C++ adapter, no data/network required.

Run: python scripts/test_k2_demo.py
"""
import array
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

BIN = Path("/tmp/earth-index-k2-upstream/build/bin/earth_index_demo")


def exercise(values, rows, cols, typ):
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        (tmp/"raw").write_bytes(array.array("i",values).tobytes())
        (tmp/"queries").write_text(f"0 {rows-1} 0 {cols-1} 1 100\n7 30 9 30 0 65535\n0 {rows-1} 0 {cols-1} 65535 65535\n")
        return subprocess.run([str(BIN),str(tmp/"raw"),str(rows),str(cols),str(tmp/"out"),str(tmp/"queries"),str(typ)],
                              capture_output=True,text=True)


class CodecChecks(unittest.TestCase):
    def test_random_rectangular_with_missing_and_extremes(self):
        rng=random.Random(73)
        values=[rng.choice([0,1,2,50,100,65535]) for _ in range(63*45)]
        for typ in (1,11):
            with self.subTest(typ=typ):
                r=exercise(values,63,45,typ)
                self.assertEqual(r.returncode,0,r.stderr)
                result=json.loads(r.stdout)
                expected=[i for i,v in enumerate(values) if 1<=v<=100]
                self.assertEqual(result["queries"][0]["cells"],expected)
                self.assertEqual(result["pixels_verified"],len(values))

    def test_known_upstream_uniform_window_defect(self):
        # The unmodified upstream root-uniform get_values_window branch starts
        # its column loop at yend instead of yini. Keep the failure visible;
        # this demo deliberately does not silently patch query algorithms.
        for rows,cols in ((64,64),(63,45)):
            for typ in (1,11):
                r=exercise([50]*(rows*cols),rows,cols,typ)
                self.assertNotEqual(r.returncode,0)
                self.assertIn("window mismatch",r.stderr)


if __name__=="__main__": unittest.main()
