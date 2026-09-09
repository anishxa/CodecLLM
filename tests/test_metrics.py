import unittest
import numpy as np
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics.endpointing import (
    compute_cutoff_rate,
    compute_endpoint_latency,
    compute_headline_disparity
)

class TestEndpointingMetricsKnownAnswers(unittest.TestCase):
    """
    Enforces Rule 7: Tests check correctness, not shapes.
    Uses hand-computed known answer fixtures.
    """
    def setUp(self):
        # Hand-computed fixture:
        # eot:  [900, 1200, 1500, 800, 2000]
        # eos:  [1000, 1000, 1000, 1000, 1000]
        # Cutoffs: 900 < 1000 (Cutoff), 800 < 1000 (Cutoff) -> 2 / 5 = 0.40
        # Non-cutoffs latencies: [200, 500, 1000]
        # Median latency: 500.0 ms
        # 90th percentile latency: 900.0 ms
        self.eot = np.array([900, 1200, 1500, 800, 2000])
        self.eos = np.array([1000, 1000, 1000, 1000, 1000])

    def test_cutoff_rate_known_answer(self):
        expected_cutoff_rate = 0.40
        actual_cutoff_rate = compute_cutoff_rate(self.eot, self.eos)
        self.assertAlmostEqual(actual_cutoff_rate, expected_cutoff_rate, places=6)

    def test_endpoint_latency_known_answer(self):
        expected_median = 500.0
        expected_p90 = 900.0
        
        actual_latency = compute_endpoint_latency(self.eot, self.eos)
        self.assertAlmostEqual(actual_latency["median_ms"], expected_median, places=4)
        self.assertAlmostEqual(actual_latency["p90_ms"], expected_p90, places=4)

    def test_headline_disparity_known_answer(self):
        dysfluent_cutoff = 0.45
        fluent_cutoff = 0.05
        expected_disparity = 0.40
        
        actual_disparity = compute_headline_disparity(dysfluent_cutoff, fluent_cutoff)
        self.assertAlmostEqual(actual_disparity, expected_disparity, places=6)

if __name__ == "__main__":
    unittest.main()
