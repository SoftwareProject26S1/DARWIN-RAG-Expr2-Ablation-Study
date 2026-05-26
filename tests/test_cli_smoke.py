from contextlib import redirect_stdout
from io import StringIO
from importlib.util import find_spec
from importlib import import_module
import unittest


class CliSmokeTest(unittest.TestCase):
    def test_main_reports_phase_one_scaffold(self) -> None:
        self.assertIsNotNone(find_spec("darwin_rag_exp2"))
        main = import_module("darwin_rag_exp2.cli").main
        stdout = StringIO()

        with redirect_stdout(stdout):
            result = main([])

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "DARWIN-RAG Exp2: Phase 1 scaffold ready.\n")


if __name__ == "__main__":
    unittest.main()
