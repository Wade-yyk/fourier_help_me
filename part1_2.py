import argparse
import json
from pathlib import Path
from string import Template

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent


def finite_differences(image, threshold):
    dx = convolve2d(image, [[1, 0, -1]], mode="same", boundary="symm")
    dy = convolve2d(image, [[1], [0], [-1]], mode="same", boundary="symm")
    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    edges = magnitude > threshold
    return dx, dy, magnitude, edges


def save_image(path, values):
    Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(path, optimize=True)


def render_section(metrics):
    return Template((ROOT / "templates" / "part1_2.html").read_text(encoding="utf-8")).substitute(
        width=metrics["shape_hw"][1], height=metrics["shape_hw"][0],
        threshold=f'{metrics["threshold"]:g}',
        low_threshold=f'{metrics["threshold_trials"][0]["threshold"]:g}',
        high_threshold=f'{metrics["threshold_trials"][-1]["threshold"]:g}',
    )


def main():
    parser = argparse.ArgumentParser(description="Generate the Part 1.2 finite-difference results")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or not 0 < args.threshold < np.sqrt(2):
        parser.error("--threshold must be finite and between 0 and sqrt(2)")
    images_dir = ROOT / "assets" / "images" / "part1_2"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with Image.open(ROOT / "data" / "cameraman.png") as source:
        image = np.asarray(source.convert("L"), dtype=np.float64) / 255.0
    dx, dy, magnitude, edges = finite_differences(image, args.threshold)
    signed_limit = max(float(np.percentile(np.concatenate((np.abs(dx).ravel(), np.abs(dy).ravel())), 99)), 1 / 255)
    magnitude_limit = max(float(magnitude.max()), 1 / 255)
    for name, values in (("original", image), ("dx", 0.5 + dx / (2 * signed_limit)),
                         ("dy", 0.5 + dy / (2 * signed_limit)),
                         ("magnitude", magnitude / magnitude_limit), ("edges", edges)):
        save_image(images_dir / f"{name}.png", values)
    trial_thresholds = sorted(set((0.08, args.threshold, 0.30)))
    trials = []
    for threshold in trial_thresholds:
        trial_edges = magnitude > threshold
        filename = f'threshold_{threshold:g}.png'
        save_image(images_dir / filename, trial_edges)
        trials.append({"threshold": threshold, "edge_fraction": float(trial_edges.mean()), "file": filename})
    np.savez_compressed(results_dir / "part1_2_arrays.npz", grayscale=image, dx=dx, dy=dy,
                        magnitude=magnitude, edges=edges)
    metrics = {
        "input": "data/cameraman.png", "shape_hw": list(image.shape), "dtype": "float64",
        "input_range": [0, 1], "dx_kernel": [[1, 0, -1]], "dy_kernel": [[1], [0], [-1]],
        "mode": "same", "boundary": "symm", "threshold": args.threshold,
        "threshold_applied_to": "unscaled gradient magnitude",
        "edge_fraction": float(edges.mean()), "threshold_trials": trials,
        "signed_display_limit": signed_limit, "magnitude_display_max": magnitude_limit,
    }
    (results_dir / "part1_2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    from part1_1 import render_report
    render_report(json.loads((results_dir / "part1_1_metrics.json").read_text(encoding="utf-8")))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
