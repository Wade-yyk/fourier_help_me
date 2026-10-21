import cv2
import numpy as np
from scipy.signal import convolve2d


def grayscale(image):
    image = np.asarray(image, dtype=np.float64)
    return image if image.ndim == 2 else image @ np.array([0.299, 0.587, 0.114])


def gaussian_kernel(sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    size = 2 * int(np.ceil(4 * sigma)) + 1
    return cv2.getGaussianKernel(size, sigma, cv2.CV_64F).ravel()


def blur(image, sigma):
    image = np.asarray(image, dtype=np.float64)
    if sigma == 0:
        return image.copy()
    kernel = gaussian_kernel(sigma)
    return cv2.sepFilter2D(image, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT)


def unsharp_kernel(sigma, amount):
    g = gaussian_kernel(sigma)
    kernel = -amount * np.outer(g, g)
    kernel[len(g) // 2, len(g) // 2] += 1 + amount
    return kernel


def unsharp(image, sigma, amount):
    image = np.asarray(image, dtype=np.float64)
    return image + amount * (image - blur(image, sigma))


def apply_kernel(image, kernel):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return convolve2d(image, kernel, mode="same", boundary="symm")
    return np.stack([convolve2d(image[..., channel], kernel, mode="same", boundary="symm")
                     for channel in range(image.shape[2])], axis=-1)


def similarity_transform(points, targets):
    points = np.asarray(points, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    matrix = []
    values = []
    for (x, y), (u, v) in zip(points, targets):
        matrix.extend(([x, -y, 1, 0], [y, x, 0, 1]))
        values.extend((u, v))
    a, b, tx, ty = np.linalg.solve(np.asarray(matrix), np.asarray(values))
    return np.array([[a, -b, tx], [b, a, ty]])


def align_image(image, points, targets, size=384):
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape[:2]
    transform = similarity_transform(np.asarray(points) * [w, h], np.asarray(targets) * [size, size])
    aligned = cv2.warpAffine(image, transform, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return aligned, transform


def hybrid_image(low_source, high_source, sigma_low, sigma_high, gain=1.0,
                 color_low=False, color_high=False):
    low_input = low_source if color_low else grayscale(low_source)
    high_input = high_source if color_high else grayscale(high_source)
    low = blur(low_input, sigma_low)
    high = high_input - blur(high_input, sigma_high)
    if low.ndim != high.ndim:
        if low.ndim == 2:
            low = np.repeat(low[..., None], 3, axis=-1)
        else:
            high = np.repeat(high[..., None], 3, axis=-1)
    return low + gain * high, low, high


def fourier_log(image):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(grayscale(image)))))


def gaussian_stack(image, sigmas=(0, 2, 4, 8, 16, 32)):
    if sigmas[0] != 0 or any(b <= a for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("stack sigmas must start at zero and strictly increase")
    levels = [np.asarray(image, dtype=np.float64).copy()]
    for previous, current in zip(sigmas, sigmas[1:]):
        levels.append(blur(levels[-1], np.sqrt(current ** 2 - previous ** 2)))
    return np.stack(levels)


def laplacian_stack(gaussians):
    gaussians = np.asarray(gaussians, dtype=np.float64)
    return np.concatenate((gaussians[:-1] - gaussians[1:], gaussians[-1:]), axis=0)


def multiresolution_blend(image_a, image_b, mask, sigmas=(0, 2, 4, 8, 16, 32)):
    ga = gaussian_stack(image_a, sigmas)
    gb = gaussian_stack(image_b, sigmas)
    gm = gaussian_stack(mask, sigmas)
    la, lb = laplacian_stack(ga), laplacian_stack(gb)
    weights = gm[..., None] if la.ndim == 4 else gm
    weighted_a = weights * la
    weighted_b = (1 - weights) * lb
    blended_levels = weighted_a + weighted_b
    return blended_levels.sum(axis=0), {
        "ga": ga, "gb": gb, "gm": gm, "la": la, "lb": lb,
        "weighted_a": weighted_a, "weighted_b": weighted_b,
        "blended_levels": blended_levels,
    }


def psnr(reference, image):
    mse = float(np.mean((np.asarray(reference) - np.asarray(image)) ** 2))
    return float("inf") if mse == 0 else float(-10 * np.log10(mse))
