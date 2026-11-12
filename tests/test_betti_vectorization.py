"""Semantic regression for the grayscale H1 channel, using real persistence."""

import unittest

import numpy as np

from BettiVectorization import generate_betti_vectors


class BettiVectorizationTests(unittest.TestCase):
    def test_dark_ring_has_grayscale_hole(self):
        # The sublevel set contains one dark square ring before the white
        # background and its central hole enter the filtration.
        image = np.full((16, 16, 3), 255, dtype=np.uint8)
        image[3:13, 3:13] = 0
        image[6:10, 6:10] = 255

        features = generate_betti_vectors([image], n_jobs=1).to_numpy()
        self.assertEqual(features.shape, (1, 400))
        self.assertTrue(np.isfinite(features).all())
        gray_h0, gray_h1 = features[0, :50], features[0, 50:100]
        self.assertGreater(gray_h1.max(), 0, "A ring must have a persistent H1 feature")
        self.assertFalse(np.array_equal(gray_h0, gray_h1))
        # All channels are identical, so grayscale must agree with each color.
        for start in (150, 250, 350):
            np.testing.assert_array_equal(gray_h1, features[0, start:start + 50])


if __name__ == "__main__":
    unittest.main()
