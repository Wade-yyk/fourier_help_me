import argparse
import json
from pathlib import Path
from string import Template

import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from part1_2 import finite_differences, save_image

ROOT = Path(__file__).resolve().parent


def gaussian_derivatives(image, size=13, sigma=1.5):
    if size < 3 or size % 2 == 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("size must be odd and >= 3; sigma must be finite and positive")
    image = np.asarray(image, dtype=np.float64)
    g = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)
    gaussian = g @ g.T
    dog_x = convolve2d(gaussian, [[1, 0, -1]], mode="full")
    dog_y = convolve2d(gaussian, [[1], [0], [-1]], mode="full")
    halo = size // 2 + 1
    extended = np.pad(image, halo, mode="symmetric")
    crop = (slice(halo, -halo), slice(halo, -halo))
    blurred_extended = convolve2d(extended, gaussian, mode="same")
    blurred = blurred_extended[crop]
    sequential_dx = convolve2d(blurred_extended, [[1, 0, -1]], mode="same")[crop]
    sequential_dy = convolve2d(blurred_extended, [[1], [0], [-1]], mode="same")[crop]
    dog_dx = convolve2d(extended, dog_x, mode="same")[crop]
    dog_dy = convolve2d(extended, dog_y, mode="same")[crop]
    return {
        "gaussian": gaussian, "dog_x": dog_x, "dog_y": dog_y, "blurred": blurred,
        "sequential_dx": sequential_dx, "sequential_dy": sequential_dy,
        "dog_dx": dog_dx, "dog_dy": dog_dy,
        "sequential_magnitude": np.sqrt(sequential_dx ** 2 + sequential_dy ** 2),
        "dog_magnitude": np.sqrt(dog_dx ** 2 + dog_dy ** 2),
    }


def gradient_orientation(dx, dy):
    dx, dy = np.broadcast_arrays(np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64))
    if not np.isfinite(dx).all() or not np.isfinite(dy).all():
        raise ValueError("gradient components must be finite")
    ax, ay = np.abs(dx), np.abs(dy)
    larger = np.maximum(ax, ay)
    ratio = np.divide(np.minimum(ax, ay), larger, out=np.zeros_like(larger), where=larger > 0)
    reduce = ratio > np.sqrt(2) - 1
    z = np.where(reduce, (ratio - 1) / (ratio + 1), ratio)
    power = z.copy()
    base = np.zeros_like(z)
    for n in range(20):
        base += power / (2 * n + 1)
        power *= -z * z
    base += np.where(reduce, np.pi / 4, 0)
    base = np.where(ay > ax, np.pi / 2 - base, base)
    base = np.where(dx < 0, np.pi - base, base)
    base = np.where(dy < 0, -base, base)
    return np.where(larger > 0, np.mod(base, 2 * np.pi), 0)


def orientation_rgb(dx, dy, value_limit):
    theta = gradient_orientation(dx, dy)
    magnitude = np.sqrt(np.asarray(dx) ** 2 + np.asarray(dy) ** 2)
    hsv = np.stack((theta * 180 / np.pi, np.ones_like(theta),
                    np.clip(magnitude / value_limit, 0, 1)), axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), theta


def save_hue_wheel(path):
    coordinates = np.linspace(-1, 1, 256)
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.sqrt(x ** 2 + y ** 2)
    rgb, _ = orientation_rgb(x, y, 0.001)
    rgb[(radius > 0.93) | (radius < 0.35)] = 255
    Image.fromarray(rgb).save(path, optimize=True)


def render_section(metrics):
    return Template((ROOT / "templates" / "part1_3.html").read_text(encoding="utf-8")).substitute(
        size=metrics["gaussian_size"], sigma=f'{metrics["sigma"]:g}',
        threshold=f'{metrics["threshold"]:g}', error=f'{metrics["max_absolute_error"]:.2e}',
        raw_threshold=f'{metrics["part1_2_threshold"]:g}',
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Gaussian derivatives and HSV orientations")
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--size", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.12)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or not 0 < args.threshold < np.sqrt(2):
        parser.error("--threshold must be finite and between 0 and sqrt(2)")
    images_dir = ROOT / "assets" / "images" / "part1_3"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with Image.open(ROOT / "data" / "cameraman.png") as source:
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    result = gaussian_derivatives(image, args.size, args.sigma)
    error = 0.0
    for name in ("dx", "dy", "magnitude"):
        left, right = result[f"sequential_{name}"], result[f"dog_{name}"]
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
        error = max(error, float(np.max(np.abs(left - right))))
    previous = json.loads((results_dir / "part1_2_metrics.json").read_text(encoding="utf-8"))
    raw_threshold = previous["threshold"]
    raw_dx, raw_dy, raw_magnitude, raw_edges = finite_differences(image, raw_threshold)
    signed_limit = max(float(np.percentile(np.concatenate((np.abs(raw_dx).ravel(), np.abs(raw_dy).ravel())), 99)), 1 / 255)
    magnitude_limit = float(raw_magnitude.max())
    edges = result["sequential_magnitude"] > args.threshold
    result["edges"] = edges
    for name, values in (
        ("blurred", result["blurred"]),
        ("dx", 0.5 + result["sequential_dx"] / (2 * signed_limit)),
        ("dy", 0.5 + result["sequential_dy"] / (2 * signed_limit)),
        ("magnitude", result["sequential_magnitude"] / magnitude_limit),
        ("dog_magnitude", result["dog_magnitude"] / magnitude_limit),
        ("edges", edges),
        ("gaussian", result["gaussian"] / result["gaussian"].max()),
    ):
        save_image(images_dir / f"{name}.png", values)
    for name in ("dog_x", "dog_y"):
        save_image(images_dir / f"{name}.png", 0.5 + result[name] / (2 * np.abs(result[name]).max()))
    orientation_limit = max(float(np.percentile(raw_magnitude, 99)), 1 / 255)
    raw_rgb, raw_theta = orientation_rgb(raw_dx, raw_dy, orientation_limit)
    smooth_rgb, smooth_theta = orientation_rgb(result["dog_dx"], result["dog_dy"], orientation_limit)
    Image.fromarray(raw_rgb).save(images_dir / "orientation_raw.png", optimize=True)
    Image.fromarray(smooth_rgb).save(images_dir / "orientation_smooth.png", optimize=True)
    save_hue_wheel(images_dir / "hue_wheel.png")
    np.savez_compressed(results_dir / "part1_3_arrays.npz", grayscale=image,
                        orientation_raw=raw_theta, orientation_smooth=smooth_theta, **result)
    trials = []
    for threshold in sorted(set((0.05, 0.08, args.threshold))):
        trial_edges = result["sequential_magnitude"] > threshold
        save_image(images_dir / f"threshold_{threshold:g}.png", trial_edges)
        trials.append({"threshold": threshold, "edge_fraction": float(trial_edges.mean())})
    metrics = {
        "input": "data/cameraman.png", "shape_hw": list(image.shape),
        "gaussian_size": args.size, "sigma": args.sigma, "opencv": cv2.__version__,
        "gaussian_sum": float(result["gaussian"].sum()),
        "dog_x_shape": list(result["dog_x"].shape), "dog_y_shape": list(result["dog_y"].shape),
        "threshold": args.threshold, "part1_2_threshold": raw_threshold,
        "edge_fraction": float(edges.mean()), "threshold_trials": trials,
        "max_absolute_error": error, "comparison_region": "entire image including borders",
        "boundary": "one symmetric extension by Gaussian radius plus derivative radius; crop after filtering",
        "derivative_display_limit": signed_limit, "magnitude_display_max": magnitude_limit,
        "orientation": {"method": "octant folding, range reduction, 20-term alternating series, quadrant restoration",
                        "range": "[0, 2*pi)", "hsv": "H=degrees(theta), S=1, V=clip(magnitude/shared_limit,0,1)",
                        "shared_value_limit": orientation_limit, "positive_y": "down in image coordinates",
                        "zero_gradient": "theta=0, V=0 (black)"},
    }
    (results_dir / "part1_3_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    from part1_1 import render_report
    render_report(json.loads((results_dir / "part1_1_metrics.json").read_text(encoding="utf-8")))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
