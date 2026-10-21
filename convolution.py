import numpy as np


def _prepare(image, kernel, mode):
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2 or not image.size or not kernel.size:
        raise ValueError("image and kernel must be nonempty 2-D arrays")
    if not np.isfinite(image).all() or not np.isfinite(kernel).all():
        raise ValueError("image and kernel must contain finite values")
    h, w = image.shape
    kh, kw = kernel.shape
    if mode == "same":

        padding = ((kh // 2, (kh - 1) // 2), (kw // 2, (kw - 1) // 2))
        shape = (h, w)
    elif mode == "full":
        padding = ((kh - 1, kh - 1), (kw - 1, kw - 1))
        shape = (h + kh - 1, w + kw - 1)
    elif mode == "valid":
        if kh > h or kw > w:
            raise ValueError("valid mode requires the kernel to fit inside the image")
        padding = ((0, 0), (0, 0))
        shape = (h - kh + 1, w - kw + 1)
    else:
        raise ValueError("mode must be 'same', 'full', or 'valid'")
    padded = np.pad(image, padding, mode="constant", constant_values=0)

    flipped = kernel[::-1, ::-1]
    return padded, flipped, np.zeros(shape, dtype=np.float64)


def convolve_four_loops(image, kernel, mode="same"):
    padded, flipped, output = _prepare(image, kernel, mode)
    kh, kw = flipped.shape
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            for kr in range(kh):
                for kc in range(kw):
                    output[row, col] += padded[row + kr, col + kc] * flipped[kr, kc]
    return output


def convolve_two_loops(image, kernel, mode="same"):
    padded, flipped, output = _prepare(image, kernel, mode)
    kh, kw = flipped.shape
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            window = padded[row:row + kh, col:col + kw]
            output[row, col] = np.sum(window * flipped)
    return output
