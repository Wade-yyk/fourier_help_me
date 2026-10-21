import unittest

import numpy as np
from scipy.signal import convolve2d

from part1_3 import gaussian_derivatives, gradient_orientation, orientation_rgb


class GaussianGradientTests(unittest.TestCase):
    def test_kernels_and_full_image_equivalence(self):
        image = np.random.default_rng(671).random((17, 23))
        result = gaussian_derivatives(image, size=7, sigma=1.2)
        self.assertAlmostEqual(result["gaussian"].sum(), 1)
        self.assertEqual(result["dog_x"].shape, (7, 9))
        self.assertEqual(result["dog_y"].shape, (9, 7))
        np.testing.assert_allclose(result["dog_x"], -result["dog_x"][:, ::-1], atol=1e-15)
        np.testing.assert_allclose(result["dog_y"], -result["dog_y"][::-1, :], atol=1e-15)
        np.testing.assert_allclose(result["blurred"], convolve2d(image, result["gaussian"],
                                                             mode="same", boundary="symm"), atol=1e-14)
        for name in ("dx", "dy", "magnitude"):
            np.testing.assert_allclose(result[f"sequential_{name}"], result[f"dog_{name}"], atol=1e-14)

    def test_constant_image_and_invalid_gaussian(self):
        result = gaussian_derivatives(np.ones((11, 15)), size=7, sigma=1.2)
        np.testing.assert_allclose(result["dog_magnitude"], 0, atol=1e-15)
        for size, sigma in ((2, 1), (4, 1), (7, 0), (7, float("nan"))):
            with self.assertRaises(ValueError):
                gaussian_derivatives(np.ones((5, 5)), size, sigma)

    def test_orientation_axes_quadrants_and_zero(self):
        dx = np.array([1, 1, 0, -1, -1, -1, 0, 1, 0], dtype=float)
        dy = np.array([0, 1, 1, 1, 0, -1, -1, -1, 0], dtype=float)
        expected = np.append(np.arange(8) * np.pi / 4, 0)
        np.testing.assert_allclose(gradient_orientation(dx, dy), expected, atol=1e-14)

    def test_orientation_reconstructs_random_unit_vectors(self):
        rng = np.random.default_rng(671)
        dx, dy = rng.normal(size=(2, 10000))
        theta = gradient_orientation(dx, dy)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        np.testing.assert_allclose(np.cos(theta), dx / magnitude, atol=2e-14)
        np.testing.assert_allclose(np.sin(theta), dy / magnitude, atol=2e-14)
        self.assertTrue(np.all((theta >= 0) & (theta < 2 * np.pi)))

    def test_hsv_primary_directions_and_flat_black(self):
        rgb, _ = orientation_rgb(np.array([[1, 0, -1, 0, 0.0]]),
                                 np.array([[0, 1, 0, -1, 0.0]]), 1)
        np.testing.assert_allclose(rgb[0], [[255, 0, 0], [128, 255, 0], [0, 255, 255],
                                          [128, 0, 255], [0, 0, 0]], atol=1)


if __name__ == "__main__":
    unittest.main()
