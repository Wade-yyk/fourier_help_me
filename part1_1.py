import argparse
import html
import inspect
import json
import platform
from pathlib import Path
from string import Template
from time import perf_counter

import numpy as np
import PIL
from PIL import Image, ImageOps
import scipy
from scipy.signal import convolve2d

import convolution

ROOT = Path(__file__).resolve().parent
DEFAULT_PHOTO = ROOT / "mypic" / "2b81a35166f5ac3836f246ce1fa9601b.jpg"


def measure(function, image, kernel, repeats):
    function(image, kernel)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = function(image, kernel)
        samples.append(perf_counter() - start)
    return result, float(np.median(samples)), samples


def scipy_reference(image, kernel):
    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


def save_grayscale(path, values):
    pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, optimize=True)


def render_report(measurements):
    height, width = measurements["processed_shape_hw"]
    filters = measurements["filters"]
    rows = []
    for key, label in (("box", "9 × 9 box"), ("dx", "Dx"), ("dy", "Dy")):
        timings = filters[key]["measurements"]
        rows.append(
            f'<tr><th scope="row">{label}</th>'
            + "".join(f'<td>{timings[m]["median_seconds"] * 1000:.3f}</td>'
                      for m in ("four", "two", "scipy"))
            + f'<td>{timings["four"]["max_absolute_error"]:.2e}</td>'
            + f'<td>{timings["two"]["max_absolute_error"]:.2e}</td></tr>'
        )
    timings = filters["box"]["measurements"]
    box_speedup = timings["four"]["median_seconds"] / timings["two"]["median_seconds"]
    finite_note = (
        "For the three-tap derivatives, NumPy's per-window overhead makes two loops slower here."
        if all(filters[k]["measurements"]["two"]["median_seconds"] >
               filters[k]["measurements"]["four"]["median_seconds"] for k in ("dx", "dy"))
        else "For three-tap derivatives, per-window NumPy overhead affects the speedup."
    )
    part1_2_section = ""
    part1_2_metrics = ROOT / "results" / "part1_2_metrics.json"
    if part1_2_metrics.exists():
        from part1_2 import render_section
        part1_2_section = render_section(json.loads(part1_2_metrics.read_text(encoding="utf-8")))
    part1_3_section = ""
    part1_3_metrics = ROOT / "results" / "part1_3_metrics.json"
    if part1_3_metrics.exists():
        from part1_3 import render_section as render_gaussian_section
        part1_3_section = render_gaussian_section(json.loads(part1_3_metrics.read_text(encoding="utf-8")))
    part2_section = ""
    part2_metrics = ROOT / "results" / "part2_metrics.json"
    if part2_metrics.exists():
        from part2 import render_section as render_frequency_section
        part2_section = render_frequency_section(json.loads(part2_metrics.read_text(encoding="utf-8")))
    template = Template((ROOT / "templates" / "part1_1.html").read_text(encoding="utf-8"))
    report = template.substitute(
        width=width, height=height, rows="\n".join(rows), repeats=measurements["repeats"],
        signed_limit=f'{measurements["signed_display_limit"]:.4f}',
        box_speedup=f"{box_speedup:.1f}", finite_note=finite_note,
        four_code=html.escape(inspect.getsource(convolution.convolve_four_loops).strip()),
        two_code=html.escape(inspect.getsource(convolution.convolve_two_loops).strip()),
        part1_2_section=part1_2_section,
        part1_3_section=part1_3_section,
        part2_section=part2_section,
    )
    (ROOT / "index.html").write_text(report, encoding="utf-8")
    print(f"Generated {ROOT / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description="Generate the Part 1.1 images, measurements, and report")
    parser.add_argument("--image", type=Path, default=DEFAULT_PHOTO)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate HTML from existing metrics without rerunning filters")
    args = parser.parse_args()
    if args.report_only:
        render_report(json.loads((ROOT / "results" / "part1_1_metrics.json").read_text(encoding="utf-8")))
        return
    if args.max_side < 9 or args.repeats < 1:
        parser.error("--max-side must be >= 9 and --repeats must be >= 1")

    images_dir = ROOT / "assets" / "images" / "part1_1"
    results_dir = ROOT / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    with Image.open(args.image) as source:
        upright = ImageOps.exif_transpose(source)
        original_size = upright.size
        gray = upright.convert("L")
        gray.thumbnail((args.max_side, args.max_side), Image.Resampling.LANCZOS)
        image = np.asarray(gray, dtype=np.float64) / 255.0
    save_grayscale(images_dir / "grayscale.png", image)


    kernels = {
        "box": np.ones((9, 9), dtype=np.float64) / 81.0,
        "dx": np.array([[1.0, 0.0, -1.0]]),
        "dy": np.array([[1.0], [0.0], [-1.0]]),
    }
    methods = {
        "four": convolution.convolve_four_loops,
        "two": convolution.convolve_two_loops,
        "scipy": scipy_reference,
    }
    results = {}
    for key, kernel in kernels.items():
        outputs, timings = {}, {}
        for method, function in methods.items():
            output, median, samples = measure(function, image, kernel, args.repeats)
            outputs[method] = output
            timings[method] = {"median_seconds": median, "samples_seconds": samples}
        for method in ("four", "two"):
            np.testing.assert_allclose(outputs[method], outputs["scipy"], rtol=1e-12, atol=1e-12)
            timings[method]["max_absolute_error"] = float(np.max(np.abs(outputs[method] - outputs["scipy"])))
        results[key] = {"kernel": kernel.tolist(), "outputs": outputs, "measurements": timings}
        print(key, json.dumps(timings))



    derivative_values = np.concatenate([np.abs(results[k]["outputs"]["scipy"]).ravel() for k in ("dx", "dy")])
    signed_limit = max(float(np.percentile(derivative_values, 99)), 1 / 255)
    for key, result in results.items():
        for method, output in result["outputs"].items():
            display = output if key == "box" else 0.5 + output / (2 * signed_limit)
            save_grayscale(images_dir / f"{key}_{method}.png", display)


    np.savez_compressed(results_dir / "part1_1_arrays.npz", grayscale=image,
                        **{f"{key}_{method}": output for key, item in results.items()
                           for method, output in item["outputs"].items()})
    measurements = {
        "input": str(args.image.resolve().relative_to(ROOT)) if args.image.resolve().is_relative_to(ROOT) else args.image.name,
        "original_size_wh": original_size, "processed_shape_hw": image.shape,
        "dtype": str(image.dtype), "mode": "same", "boundary": "fill", "fillvalue": 0,
        "repeats": args.repeats, "warmups_per_method_kernel": 1,
        "signed_display_limit": signed_limit,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "pillow": PIL.__version__},
        "filters": {key: {"kernel": item["kernel"], "measurements": item["measurements"]}
                    for key, item in results.items()},
    }
    (results_dir / "part1_1_metrics.json").write_text(json.dumps(measurements, indent=2) + "\n", encoding="utf-8")

    render_report(measurements)


if __name__ == "__main__":
    main()
