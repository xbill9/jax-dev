"""PyO3 binding tests that need no TPU: the module imports, reports its build, and rejects bad
arguments with Python exceptions instead of panics. Build first with `make rust`.

The chip-side checks (selftest, a full Engine, parity against the JAX engine) are
tools/parity.py, run on the TPU VM.
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "rust", "build")
sys.path.insert(0, BUILD)

try:
    import jaxrust_rs
except ImportError as e:  # not built
    jaxrust_rs = None
    REASON = f"jaxrust_rs not built ({e}); run `make rust`"


@unittest.skipIf(jaxrust_rs is None, "jaxrust_rs not built; run `make rust`")
class BindingTests(unittest.TestCase):
    def test_build_metadata(self):
        self.assertEqual(set(jaxrust_rs.BUILD), {"rustc", "pjrt_header", "crate"})
        self.assertEqual(jaxrust_rs.BUILD["pjrt_header"], "0.117")
        self.assertEqual(jaxrust_rs.BUCKET, 128)

    def test_exports_match_stub(self):
        stub = open(os.path.join(ROOT, "rust", "python", "jaxrust_rs.pyi")).read()
        for name in ("init", "selftest", "emit", "Engine", "DecodeResult", "BUCKET", "BUILD"):
            self.assertTrue(hasattr(jaxrust_rs, name), name)
            self.assertIn(name, stub)

    def test_init_bad_plugin_is_runtime_error(self):
        with self.assertRaises(RuntimeError) as cm:
            jaxrust_rs.init("/nonexistent/libtpu.so")
        self.assertIn("dlopen", str(cm.exception))

    def test_engine_rejects_bad_context(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (0, -128, 100):
                with self.assertRaises(ValueError):
                    jaxrust_rs.Engine(d, bad)

    def test_engine_rejects_missing_dir(self):
        with self.assertRaises(ValueError):
            jaxrust_rs.Engine("/nonexistent/model", 256)

    def test_emit_rejects_unknown_kind(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                jaxrust_rs.emit(d, "train", 128, 256)


if __name__ == "__main__":
    unittest.main()
