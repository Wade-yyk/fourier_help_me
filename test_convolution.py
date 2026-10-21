import unittest

import numpy as np
from scipy.signal import convolve2d

from convolution import convolve_four_loops, convolve_two_loops


class ConvolutionTests(unittest.TestCase):
    implementations = (convolve_four_loops, convolve_two_loops)

    def test_scipy_parity(self):
        rng = np.random.default_rng(671)

        for image_shape in ((1, 1), (4, 7), (6, 5)):
            image = rng.normal(size=image_shape)
            for kernel_shape in ((1, 1), (1, 3), (3, 1), (2, 4), (3, 3), (7, 8)):
                kernel = rng.normal(size=kernel_shape)
                for mode in ("same", "full", "valid"):
                    if mode == "valid" and any(k > n for k, n in zip(kernel_shape, image_shape)):
                        continue
                    expected = convolve2d(image, kernel, mode=mode, boundary="fill", fillvalue=0)
                    for implementation in self.implementations:
                        with self.subTest(shape=image_shape, kernel=kernel_shape, mode=mode,
                                          implementation=implementation.__name__):
                            np.testing.assert_allclose(implementation(image, kernel, mode),
                                                       expected, atol=1e-12, rtol=1e-12)

    def test_zero_padding_does_not_renormalize_box(self):
        for implementation in self.implementations:
            result = implementation(np.ones((5, 5)), np.ones((3, 3)) / 9)
            self.assertAlmostEqual(result[0, 0], 4 / 9)
            self.assertAlmostEqual(result[2, 2], 1)

    def test_derivative_sign_and_axis(self):
        ramp = np.tile(np.arange(6), (5, 1))
        for implementation in self.implementations:
            dx = implementation(ramp, [[1, 0, -1]])
            dy = implementation(ramp, [[1], [0], [-1]])
            np.testing.assert_array_equal(dx[1:-1, 1:-1], 2)
            np.testing.assert_array_equal(dy[1:-1, 1:-1], 0)

    def test_uint8_is_converted_before_arithmetic(self):
        for implementation in self.implementations:
            result = implementation(np.array([[255, 0, 255]], dtype=np.uint8), [[1, 0, -1]])
            self.assertEqual(result.dtype, np.float64)
            np.testing.assert_array_equal(result, [[0, 0, 0]])
            result = implementation(np.array([[255, 0, 0]], dtype=np.uint8), [[1, 0, -1]])
            self.assertEqual(result[0, 1], -255)

    def test_invalid_inputs(self):
        for implementation in self.implementations:
            for image, kernel, mode in (([], [[1]], "same"), ([1, 2], [[1]], "same"),
                                        ([[1]], [[]], "same"), ([[np.nan]], [[1]], "same"),
                                        ([[1]], [[1]], "bad"), ([[1]], np.ones((2, 2)), "valid")):
                with self.assertRaises(ValueError):
                    implementation(image, kernel, mode)


if __name__ == "__main__":
    unittest.main()
