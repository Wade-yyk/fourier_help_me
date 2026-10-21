import html
import json
from pathlib import Path
from string import Template

import cv2
import numpy as np
from PIL import Image, ImageOps

from frequency import (align_image, apply_kernel, blur, fourier_log, gaussian_stack,
                       grayscale, hybrid_image, laplacian_stack, multiresolution_blend,
                       psnr, unsharp, unsharp_kernel)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "part2"
IMAGES = ROOT / "assets" / "images" / "part2"
SIGMAS = (0, 2, 4, 8, 16, 32)


def load_image(path, max_side=384):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255


def save(name, values, signed=False, scale=None):
    values = np.asarray(values)
    if signed:
        limit = max(float(np.percentile(np.abs(values), 99)) if scale is None else float(scale), 1e-12)
        values = 0.5 + values / (2 * limit)
    path = IMAGES / f"{name}.png"
    pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path, optimize=True)
    return f"assets/images/part2/{name}.png"


def figure(path, caption, small=False):
    image = f'<img src="{path}" alt="{html.escape(caption, quote=True)}" loading="lazy">'
    if small:
        image = f'<div class="small-preview">{image}</div>'
    return f'<figure>{image}<figcaption>{html.escape(caption)}</figcaption></figure>'


def gallery(items, columns=4):
    return f'<div class="image-grid part2-grid cols-{columns}">' + "".join(items) + '</div>'


def save_figure(name, values, caption, **kwargs):
    return figure(save(name, values, **kwargs), caption)


def save_far(name, values):
    pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    preview = Image.fromarray(pixels).resize((48, 48), Image.Resampling.LANCZOS)
    path = IMAGES / f"{name}_far.png"
    preview.save(path, optimize=True)
    return figure(f"assets/images/part2/{name}_far.png", "Small / far view", small=True)


def sharpening_results():
    sections = []
    errors = []
    for key, label, path in (("taj", "Taj Mahal", DATA / "taj.jpg"),
                              ("portrait", "My portrait", ROOT / "mypic" / "2b81a35166f5ac3836f246ce1fa9601b.jpg")):
        original = load_image(path)
        blurred = blur(original, 1.5)
        high = original - blurred
        items = [save_figure(f"sharp_{key}_original", original, "Original"),
                 save_figure(f"sharp_{key}_blur", blurred, "Gaussian blur · σ = 1.5"),
                 save_figure(f"sharp_{key}_high", high, "High frequencies · zero = gray", signed=True)]
        for amount in (0.5, 1, 2):
            result = apply_kernel(original, unsharp_kernel(1.5, amount))
            expected = unsharp(original, 1.5, amount)
            np.testing.assert_allclose(result, expected, atol=2e-13, rtol=2e-13)
            errors.append(float(np.max(np.abs(result - expected))))
            items.append(save_figure(f"sharp_{key}_{amount:g}", result, f"Sharpened · α = {amount:g}"))
        sections.append(f'<h3>{label}</h3>' + gallery(items, 6))
    reference = load_image(DATA / "flower.jpg")
    degraded = blur(reference, 2)
    restored = np.clip(unsharp(degraded, 2, 2), 0, 1)
    evaluation = gallery([
        save_figure("sharp_eval_reference", reference, "Sharp reference"),
        save_figure("sharp_eval_blur", degraded, "Artificial blur · σ = 2"),
        save_figure("sharp_eval_restored", restored, "Sharpened again · α = 2"),
    ], 3)
    return {"sharpening": "".join(sections), "evaluation": evaluation,
            "blur_psnr": f"{psnr(reference, degraded):.2f}",
            "sharp_psnr": f"{psnr(reference, restored):.2f}"}, {
        "sigma": 1.5, "amounts": [0.5, 1, 2], "single_kernel_max_error": max(errors),
        "evaluation_blur_sigma": 2, "evaluation_alpha": 2,
        "evaluation_psnr_blurred": psnr(reference, degraded),
        "evaluation_psnr_resharpened": psnr(reference, restored),
    }


def prepare_pairs():
    eye_targets = ((0.34, 0.38), (0.66, 0.38))
    sources = {
        "derek": ("DerekPicture.jpg", ((298 / 732, 344 / 1024), (439 / 732, 329 / 1024))),
        "nutmeg": ("nutmeg.jpg", ((610 / 1408, 282 / 1056), (752 / 1408, 363 / 1056))),
        "ke": ("portrait_far.jpg", ((0.30, 0.435), (0.66, 0.427))),
        "leo": ("leo.jpg", ((162 / 366, 192 / 489), (239 / 366, 193 / 489))),
    }
    aligned, originals, transforms = {}, {}, {}
    for key, (filename, points) in sources.items():
        originals[key] = load_image(DATA / filename, 512)
        aligned[key], transform = align_image(originals[key], points, eye_targets)
        transforms[key] = {"source": filename, "normalized_points": points,
                           "normalized_targets": eye_targets, "matrix": transform.tolist()}
    for key, filename, points in (
        ("flower", "flower.jpg", ((0.20, 0.45), (0.80, 0.45))),
        ("orange", "orange.jpeg", ((0.20, 0.50), (0.80, 0.50))),
    ):
        originals[key] = load_image(DATA / filename, 512)
        targets = ((0.22, 0.50), (0.78, 0.50))
        aligned[key], transform = align_image(originals[key], points, targets)
        transforms[key] = {"source": filename, "normalized_points": points,
                           "normalized_targets": targets, "matrix": transform.tolist()}
    return originals, aligned, transforms


def hybrid_results(originals, aligned, transforms):
    cards = []
    checks = {}
    for key, title, a, b, sigma_low, sigma_high, gain in (
        ("derek", "Derek + Nutmeg", "derek", "nutmeg", 8, 4, 1.2),
        ("flower", "Orange + my flower", "orange", "flower", 7, 3, 1.4),
    ):
        result, low, high = hybrid_image(aligned[a], aligned[b], sigma_low, sigma_high, gain)
        items = [save_figure(f"hybrid_{key}_a", originals[a], f"Low-frequency input · {a}"),
                 save_figure(f"hybrid_{key}_b", originals[b], f"High-frequency input · {b}"),
                 save_figure(f"hybrid_{key}", result, "Hybrid · close view"), save_far(f"hybrid_{key}", result)]
        cards.append(f'<h3>{title}</h3>' + gallery(items, 4)
                     + f'<p class="experiment-note">σ low = {sigma_low}, σ high = {sigma_high}, high-frequency gain = {gain:g}. '
                       f'Look close for {b}, then step back for {a}.</p>')
        checks[key] = {"sigma_low": sigma_low, "sigma_high": sigma_high, "high_gain": gain,
                       "low_input": a, "high_input": b, "color": "grayscale"}
    a, b = aligned["leo"], aligned["ke"]
    result, low, high = hybrid_image(a, b, 6, 3, 1.2, color_low=True, color_high=False)
    inputs = gallery([save_figure("hybrid_favorite_raw_leo", originals["leo"], "Leo · before alignment"),
                      save_figure("hybrid_favorite_raw_ke", originals["ke"], "My portrait · before alignment")], 2)
    pipeline = gallery([save_figure("hybrid_favorite_aligned_leo", a, "Leo · aligned"),
                        save_figure("hybrid_favorite_aligned_ke", b, "My portrait · aligned"),
                        save_figure("hybrid_favorite_low", low, "Low frequencies · σ = 6"),
                        save_figure("hybrid_favorite_high", high, "High frequencies · σ = 3", signed=True),
                        save_figure("hybrid_favorite", result, "Hybrid · color in low frequencies"),
                        save_far("hybrid_favorite", result)], 6)
    spectra = [fourier_log(values) for values in (a, b, low, high, result)]
    limit = max(float(s.max()) for s in spectra)
    fft_gallery = gallery([save_figure(f"fft_{key}", spectrum / limit, label)
                           for key, label, spectrum in zip(
                               ("input_leo", "input_ke", "low", "high", "hybrid"),
                               ("FFT · Leo", "FFT · my portrait", "FFT · low-pass", "FFT · high-pass", "FFT · hybrid"),
                               spectra)], 5)
    colors = []
    for key, label, color_low, color_high in (
        ("gray", "Both gray", False, False), ("low", "Color in low frequencies", True, False),
        ("high", "Color in high frequencies", False, True), ("both", "Color in both", True, True),
    ):
        output, _, _ = hybrid_image(a, b, 6, 3, 1.2, color_low, color_high)
        colors.append(save_figure(f"hybrid_color_{key}", output, label))
    trials = []
    for sl, sh in ((2, 1), (6, 3), (12, 6)):
        output, _, _ = hybrid_image(a, b, sl, sh, 1.2, True, False)
        trials.append(save_figure(f"hybrid_trial_{sl}_{sh}", output, f"σ low = {sl}, σ high = {sh}"))
    checks["favorite"] = {"low_input": "leo", "high_input": "ke", "sigma_low": 6, "sigma_high": 3,
                           "high_gain": 1.2, "color_low": True, "color_high": False,
                           "half_amplitude_cutoffs_cycles_per_pixel": [float(np.sqrt(np.log(2)) / (np.sqrt(2) * np.pi * s)) for s in (6, 3)],
                           "fft": "log1p(abs(fftshift(fft2(grayscale)))); shared display maximum",
                           "alignment": transforms, "cutoff_trials": [[2, 1], [6, 3], [12, 6]]}
    return {"hybrid_examples": "".join(cards), "favorite_originals": inputs,
            "hybrid_pipeline": pipeline, "hybrid_fft": fft_gallery,
            "hybrid_colors": gallery(colors, 4), "hybrid_trials": gallery(trials, 3)}, checks


def stack_grid(key, label, stack, signed):
    figures = []
    for level, image in enumerate(stack):
        caption = f"{label} · L{level}" + (" residual" if signed and level == len(stack) - 1 else "")
        figures.append(save_figure(f"stack_{key}_{level}", image, caption,
                                  signed=signed and level < len(stack) - 1))
    return gallery(figures, 6)


def contribution_figure(prefix, stacks, labels=("A", "B")):
    figures = []
    letter = 0
    for level in (0, 2, 4):
        values = [stacks[key][level] for key in ("weighted_a", "weighted_b", "blended_levels")]
        scale = max(float(np.abs(value).max()) for value in values)
        for column, (value, label) in enumerate(zip(values, (f"{labels[0]} contribution", f"{labels[1]} contribution", "Sum"))):
            figures.append(save_figure(f"{prefix}_{chr(97 + letter)}", value,
                                       f"({chr(97 + letter)}) {label} · L{level}", signed=True, scale=scale))
            letter += 1
    for key, label in (("weighted_a", f"{labels[0]} contributions summed"), ("weighted_b", f"{labels[1]} contributions summed"),
                       ("blended_levels", "Reconstructed blend")):
        figures.append(save_figure(f"{prefix}_{chr(97 + letter)}", stacks[key].sum(axis=0),
                                   f"({chr(97 + letter)}) {label}"))
        letter += 1
    return gallery(figures, 3)


def irregular_masks(size=384):
    t = np.linspace(-np.pi / 2, 3 * np.pi / 2, 24, endpoint=False)
    r = np.where(np.arange(24) % 2 == 0, 0.35, 0.235)
    sun_points = np.stack((0.5 + r * np.cos(t), 0.5 + r * np.sin(t)), axis=-1)
    face_points = np.array([[.28,.25],[.36,.12],[.57,.12],[.72,.25],[.75,.48],
                            [.68,.66],[.56,.80],[.44,.82],[.31,.68],[.24,.46]])
    masks = []
    for index, points in enumerate((sun_points, face_points)):
        mask = np.zeros((size, size), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(points * size).astype(np.int32)], 1)
        if index == 0:
            cv2.circle(mask, (size // 2, size // 2), round(0.24 * size), 1, -1)
        masks.append(mask.astype(np.float64))
    return masks


def blending_results(aligned):
    apple = load_image(DATA / "apple.jpeg")
    orange = load_image(DATA / "orange.jpeg")
    mask = np.zeros(apple.shape[:2]); mask[:, :mask.shape[1] // 2] = 1
    oraple, stacks = multiresolution_blend(apple, orange, mask, SIGMAS)
    direct = mask[..., None] * apple + (1 - mask[..., None]) * orange
    gray_blend, _ = multiresolution_blend(grayscale(apple), grayscale(orange), mask, SIGMAS)
    oraple_gallery = gallery([
        save_figure("oraple_apple", apple, "Apple"), save_figure("oraple_orange", orange, "Orange"),
        save_figure("oraple_direct", direct, "Direct splice"),
        save_figure("oraple_gray", gray_blend, "Grayscale blend"),
        save_figure("oraple_color", oraple, "Color oraple"),
    ], 5)
    gaussian_views, laplacian_views = [], []
    reconstruction_errors = {}
    for key, label, g, laplacian, original in (
        ("apple", "Apple", stacks["ga"], stacks["la"], apple),
        ("orange", "Orange", stacks["gb"], stacks["lb"], orange),
        ("oraple", "Oraple", gaussian_stack(oraple, SIGMAS), None, oraple),
    ):
        l = laplacian_stack(g) if laplacian is None else laplacian
        reconstruction_errors[key] = float(np.max(np.abs(l.sum(axis=0) - original)))
        np.testing.assert_allclose(l.sum(axis=0), original, atol=1e-12)
        gaussian_views.append(stack_grid(f"{key}_g", label, g, False))
        laplacian_views.append(stack_grid(f"{key}_l", label, l, True))
    masks = irregular_masks()
    examples = []
    favorite_stacks = None
    for key, title, a, b, current_mask in (
        ("flower", "Flower fruit", aligned["flower"], aligned["orange"], masks[0]),
        ("face", "My face in Leo's portrait", aligned["ke"], aligned["leo"], masks[1]),
    ):
        blend, current_stacks = multiresolution_blend(a, b, current_mask, SIGMAS)
        direct = current_mask[..., None] * a + (1 - current_mask[..., None]) * b
        mask_label = "Sun-shaped mask" if key == "flower" else "Face-shaped mask"
        examples.append(f'<h3>{title}</h3>' + gallery([
            save_figure(f"blend_{key}_a", a, "Input A"), save_figure(f"blend_{key}_b", b, "Input B"),
            save_figure(f"blend_{key}_mask", current_mask, f"{mask_label} · white selects A"),
            save_figure(f"blend_{key}_direct", direct, "Direct paste"),
            save_figure(f"blend_{key}", blend, "Multiresolution blend"),
        ], 5))
        if key == "face":
            favorite_stacks = current_stacks
    mask_views = stack_grid("face_mask", "Face mask", favorite_stacks["gm"], False)
    favorite_gray = gallery([save_figure("blend_face_gray", grayscale(favorite_stacks["blended_levels"].sum(axis=0)),
                                       "Reconstructed blend · grayscale")], 3)
    return {
        "oraple": oraple_gallery, "gaussian_stacks": "".join(gaussian_views),
        "laplacian_stacks": "".join(laplacian_views),
        "figure342": contribution_figure("fig342", stacks),
        "custom_blends": "".join(examples), "favorite_process": contribution_figure("blend_process", favorite_stacks, ("My portrait", "Leo")),
        "mask_stack": mask_views,
        "favorite_gray": favorite_gray,
    }, {
        "stack_sigmas": list(SIGMAS), "level_count": len(SIGMAS), "downsampling": False,
        "reconstruction_max_errors": reconstruction_errors, "figure342_levels": [0, 2, 4],
        "figure342_bottom_row": "sum of all weighted Laplacian levels for A, B, and blend",
        "irregular_masks": ["12-ray sun polygon with circular center", "hand-defined face polygon"],
        "favorite_blend": "My face in Leo's portrait", "favorite_blend_key": "face",
        "favorite_reconstruction_versions": ["RGB", "grayscale"],
        "color": "RGB channels filtered independently with the same Gaussian mask weights",
    }


def render_section(metrics):
    return Template((ROOT / "templates" / "part2.html").read_text(encoding="utf-8")).substitute(metrics["report"])


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    sharp_report, sharp_metrics = sharpening_results()
    originals, aligned, transforms = prepare_pairs()
    hybrid_report, hybrid_metrics = hybrid_results(originals, aligned, transforms)
    blend_report, blend_metrics = blending_results(aligned)
    report = {**sharp_report, **hybrid_report, **blend_report}
    metrics = {"sharpening": sharp_metrics, "hybrids": hybrid_metrics, "stacks_and_blending": blend_metrics,
               "report": report, "display": "clip to [0,1] only when exporting PNG; signed bands use gray as zero"}
    path = ROOT / "results" / "part2_metrics.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    from part1_1 import render_report
    render_report(json.loads((ROOT / "results" / "part1_1_metrics.json").read_text(encoding="utf-8")))
    print(json.dumps({"sharpening": sharp_metrics, "stacks": blend_metrics}, indent=2))


if __name__ == "__main__":
    main()
