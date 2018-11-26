"""Short tests to make sure GEMSEC can be imported."""

import unittest

from gemsec.calculation_helper import unit


class TestCalculations(unittest.TestCase):
    def test_something(self):
        self.assertEquals(1, unit(None, None, None))
