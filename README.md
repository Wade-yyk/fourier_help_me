# SYDE 671 - Assignment 2

Parts One and Two, including the mandatory color and HSV experiments, are complete locally.
Open `index.html` in a browser to see the portrait,
all filtering results, code, measured runtime/error comparisons, and explanations.
The compact Part 1.1 section includes four images, two core code snippets, one runtime/error
table, and brief explanations. The full report is in English and uses local image assets.

## Coursework notice

This project was created solely for the SYDE 671 course assignment at the University of Waterloo.
Third-party images and materials remain subject to their respective licenses and attribution requirements.

## Run

Requires Python 3.12+; commands run from this repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe part1_1.py
.\.venv\Scripts\python.exe part1_2.py
.\.venv\Scripts\python.exe part1_3.py
.\.venv\Scripts\python.exe part2.py
```

The included `.venv` is a local development environment and is ignored by Git.
Default processing preserves the portrait's aspect ratio with a maximum side of
384 pixels. Timing uses one warm-up and the median of three runs for each pair
of kernel and implementation. Use `--max-side 512 --repeats 5` to change this,
or `--image path/to/photo.jpg` to use a different portrait. Re-running overwrites
the generated Part 1.1 report, images, and metrics with the current results.
Use `python part1_1.py --report-only` to update the page from existing measurements
without rerunning the filters or changing benchmark results.
Part 1.2 appends the cameraman finite differences, gradient magnitude, and binary
edge image to the same page. Run `python part1_2.py --threshold 0.20` to regenerate
it; the script also saves threshold trials at 0.08 and 0.30 for comparison.
Part 1.1 regeneration preserves the existing Part 1.2 section.
Part 1.3 uses a 13 x 13 Gaussian with sigma 1.5 and an edge threshold of 0.12.
Run `python part1_3.py --sigma 1.5 --size 13 --threshold 0.12` to regenerate its
results and HSV views. Existing generated sections survive report regeneration.
OpenCV supplies `getGaussianKernel`; convolution uses SciPy.

The two Gaussian derivative paths share a symmetric extension of the original
image by the Gaussian radius plus one pixel. Cropping happens only after both
operations, so their outputs agree across the entire image, including borders.
The DoG kernels use full convolution to preserve their support.

Gradient orientations are computed without built-in angle functions. The code
folds the slope into the first octant, reduces its absolute value to at most
sqrt(2)-1, evaluates 20 terms of z - z^3/3 + z^5/5 - ..., and restores the quadrant.
HSV hue encodes direction, saturation is one, and value encodes magnitude; both
orientation views share the same brightness scale. Positive image y points down.

## Files

- `convolution.py`: four-loop and two-loop convolution, implemented with NumPy only.
- `part1_1.py`: image I/O, filtering, SciPy comparison, timing, and report generation.
- `part1_2.py`: cameraman derivatives, gradient magnitude, threshold trials, and page section.
- `part1_3.py`: Gaussian/DoG filters, equivalent image derivatives, and manual HSV orientations.
- `frequency.py`: unsharp masking, alignment, hybrids, FFTs, Gaussian/Laplacian stacks, and multiresolution blending.
- `part2.py`: all Part 2 inputs, experiments, visualizations, numeric checks, and page generation.
- `test_frequency.py`: unsharp-kernel equivalence, hybrid DC behavior, stack reconstruction, blend endpoints, and alignment checks.
- `templates/part2.html`: concise Part 2 descriptions and experiment galleries.
- `data/part2/sources.json`: provenance and hashes for the original inputs.
- `results/part2_metrics.json`: parameters, measured checks, and generated gallery references.
- `test_part1_3.py`: full-image equivalence, kernel symmetry, orientation quadrants, and reconstruction checks.
- `templates/part1_3.html`: concise Part 1.3 and mandatory HSV sections.
- `templates/part1_2.html`: concise Part 1.2 section template.
- `data/cameraman.png`: classic 512 x 512 cameraman image; provenance is in `data/cameraman_source.json`.
- `results/part1_2_arrays.npz`: raw signed derivatives, magnitude, and binary edges.
- `results/part1_2_metrics.json`: padding, thresholds, edge fractions, and display scales.
- `results/part1_3_arrays.npz`: Gaussian/DoG kernels, both derivative paths, edges, and orientations.
- `results/part1_3_metrics.json`: parameters, equivalence error, threshold trials, and HSV mapping.
- `test_convolution.py`: numerical, boundary, alignment, sign, and validation checks.
- `templates/part1_1.html`: editable report template; regenerate after editing.
- `index.html`: generated report; can be opened directly without a server.
- `assets/images/part1_1/`: grayscale portrait and the nine filtered result images.
- `results/part1_1_metrics.json`: all measured timings, maximum errors, and environment.
- `results/part1_1_arrays.npz`: signed float64 results, before display scaling/clipping.

The filters are a normalized 9 x 9 box and the assignment's exact finite
differences `Dx = [[1, 0, -1]]`, `Dy = [[1], [0], [-1]]`. Both implementations
flip the kernel and use zero-filled padding. `same`, `full`, and `valid` modes
are available; `valid` requires the kernel to fit within the image.

## Part Two

- 2.1: Taj Mahal and the user's portrait, blur/high-frequency components, three sharpening amounts,
  and a sharp flower -> blur -> sharpen evaluation with PSNR. Each sharpened result is computed
  with the combined unsharp kernel and checked against the separate operations.
- 2.2: Derek/Nutmeg, orange/flower, and Leo/user-portrait hybrids. The last pair includes original
  and aligned inputs, filters, five Fourier spectra, cutoff trials, and four color arrangements.
  The small previews are 48 x 48 anti-aliased reductions to illustrate distant viewing.
- 2.3: six same-size Gaussian/Laplacian levels for apple, orange, and oraple. Full stacks and the
  recreated Figure 3.42 (a)-(l) are expanded by default in the webpage. The first three figure rows use
  levels 0, 2, and 4; the final row sums all six weighted bands.
- 2.4: RGB and grayscale oraples with a straight seam, plus a cartoon sun-shaped mask for
  flower/orange and a face-shaped mask for user/Leo. The favorite user/Leo blend includes its mask
  stack, masked bands, and RGB/grayscale reconstructions. All image groups are expanded by default.

The stack sigmas are 0, 2, 4, 8, 16, 32. Every level keeps the input dimensions; no built-in
pyramid or downsampling function is used. Gaussian filtering is separable, uses reflected
boundaries, and never mixes color channels. The last Laplacian level is the Gaussian residual,
which makes reconstruction a direct sum. Mask weights add to one at every level.

Signed high-frequency images use gray as zero and contrast scaling only for display. Figure
3.42 and the favorite blend use a shared signed scale within each row. Sharpened/hybrid/blended
values are clipped only on PNG export. FFT views use log1p magnitude and a common scale.

Official public course archives supply Taj Mahal, Derek/Nutmeg, and apple/orange. Custom
experiments reuse the user's portrait and the flower/Leo assets already present in the user's
photos project. No Learn login is needed. The user's existing Part One copy is preserved.

## Leonardo DiCaprio image credit

[Leonardo DiCaprio.jpeg](https://commons.wikimedia.org/wiki/File:Leonardo_DiCaprio.jpeg)
by Jaud (Falkenauge), via Wikimedia Commons, is licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
The source photo was cropped and adjusted on Wikimedia Commons. Ke Yang's changes include
alignment, filtering, hybridization, multiresolution blending, and grayscale conversion.
Image adaptations incorporating this photograph in Parts 2.2 and 2.4 are shared under
CC BY-SA 3.0. This notice covers the photograph and image adaptations containing it.
The composites are coursework experiments; no endorsement is implied.
