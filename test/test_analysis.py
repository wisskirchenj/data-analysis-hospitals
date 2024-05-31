import unittest
from unittest.mock import patch

from hospitals.analysis import Analysis


class TestAnalysis(unittest.TestCase):

    @patch("matplotlib.pyplot.show")
    @patch("builtins.print")
    def test_answers(self, mock_print, _):
        analysis = Analysis()
        analysis.main(data_dir='./data/')
        self.assertEqual(3, mock_print.call_count)
        mock_print.assert_any_call("The answer to the 1st question: 15-35")
        mock_print.assert_any_call("The answer to the 2nd question: pregnancy")
