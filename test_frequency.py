import unittest

import numpy as np

from frequency import (apply_kernel, gaussian_stack, hybrid_image, laplacian_stack,
                       multiresolution_blend, similarity_transform, unsharp, unsharp_kernel)


class FrequencyTests(unittest.TestCase):
    def test_unsharp_single_kernel_matches_two_step(self):
        image = np.random.default_rng(2).random((21, 25, 3))
        for amount in (0, 0.5, 2):
            kernel = unsharp_kernel(1.2, amount)
            self.assertAlmostEqual(kernel.sum(), 1)
            np.testing.assert_allclose(apply_kernel(image, kernel), unsharp(image, 1.2, amount), atol=2e-14)

    def test_hybrid_constant_has_zero_high_frequencies(self):
        low = np.full((19, 23, 3), 0.3)
        high = np.full_like(low, 0.8)
        for colors in ((False, False), (True, False), (False, True), (True, True)):
            output, _, detail = hybrid_image(low, high, 2, 3, color_low=colors[0], color_high=colors[1])
            np.testing.assert_allclose(detail, 0, atol=1e-14)
            np.testing.assert_allclose(output, 0.3, atol=1e-14)

    def test_stacks_keep_size_and_reconstruct(self):
        image = np.random.default_rng(3).random((19, 23, 3))
        g = gaussian_stack(image, (0, 1, 2, 4))
        self.assertEqual(g.shape, (4, 19, 23, 3))
        np.testing.assert_allclose(laplacian_stack(g).sum(axis=0), image, atol=1e-14)

    def test_blend_endpoints_and_identical_inputs(self):
        rng = np.random.default_rng(4)
        a, b = rng.random((2, 19, 23, 3))
        for mask, expected in ((np.ones((19, 23)), a), (np.zeros((19, 23)), b)):
            result, _ = multiresolution_blend(a, b, mask, (0, 1, 2, 4))
            np.testing.assert_allclose(result, expected, atol=1e-14)
        mask = rng.random((19, 23))
        result, _ = multiresolution_blend(a, a, mask, (0, 1, 2, 4))
        np.testing.assert_allclose(result, a, atol=1e-14)

    def test_similarity_transform_matches_landmarks(self):
        points = np.array([[10, 20], [30, 35]])
        targets = np.array([[80, 100], [160, 100]])
        transform = similarity_transform(points, targets)
        mapped = np.c_[points, np.ones(2)] @ transform.T
        np.testing.assert_allclose(mapped, targets, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
