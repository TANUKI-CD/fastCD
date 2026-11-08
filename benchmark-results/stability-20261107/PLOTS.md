# Accuracy and speed charts

The six charts compare quasilinear, direct and Cariow–Cariowa multiplication for real coefficients, and quasilinear and direct multiplication for complex coefficients, at the same dimensions. Quasilinear multiplication uses balanced accumulation.

| Chart | PNG | SVG | PDF |
|:---|:---|:---|:---|
| Relative error, real coefficients | [PNG](relative-error-real.png) | [SVG](relative-error-real.svg) | [PDF](relative-error-real.pdf) |
| Speed of single products, real coefficients | [PNG](speed-real-single.png) | [SVG](speed-real-single.svg) | [PDF](speed-real-single.pdf) |
| Speed of batched products, real coefficients | [PNG](speed-real-batch.png) | [SVG](speed-real-batch.svg) | [PDF](speed-real-batch.pdf) |
| Relative error, complex coefficients | [PNG](relative-error-complex.png) | [SVG](relative-error-complex.svg) | [PDF](relative-error-complex.pdf) |
| Speed of single products, complex coefficients | [PNG](speed-complex-single.png) | [SVG](speed-complex-single.svg) | [PDF](speed-complex-single.pdf) |
| Speed of batched products, complex coefficients | [PNG](speed-complex-batch.png) | [SVG](speed-complex-batch.svg) | [PDF](speed-complex-batch.pdf) |

## Accuracy

Each point is the median relative error `E_rel` (defined in docs/stability.md) of the Gaussian and uniform cases of the accuracy study at one dimension. Products whose exact value is zero are left out; outputs with zero error are kept. Complex errors use the Euclidean norm of the coefficients. The sample keys and counts behind every point are in plot-data.json.

## Speed

Each point is the median throughput, in products per second, over 12 timing blocks run with the native benchmark runners on logical CPU 18. In every block, each algorithm is timed on the same 32 stored operand pairs per dimension, and every timing lasts at least 20 ms. For real coefficients the order of the algorithms changes from block to block. For complex coefficients, one runner process per block times single and batched products of both algorithms in rotating order, and a block's value is the median of four timings. The runners check every timed product. Bootstrap 95% intervals are in plot-data.json and plot-data.csv.

## Axes

N is on a base-two logarithmic axis. The vertical axis is logarithmic when every plotted median is positive and linear otherwise. Medians do not describe worst-case errors, and timings on this host do not predict timings on other machines.

## Reproduction

From the repository root of the machine that ran the study:

```sh
python3 build/stability-work/measured-source/benchmarks/stability_plots.py --accuracy benchmark-results/stability-20261107/raw/accuracy.json --build-dir build/stability-work/builds/balanced-simd --output benchmark-results/stability-20261107 --raw benchmark-results/stability-20261107/raw --source build/stability-work/measured-source --measure --real-algorithms quasilinear direct cariow_cariowa --blocks 12 --minimum-ms 20 --trials 32 --real-max 1024 --complex-max 512 --cpu 18
```

Without --measure, the same command redraws the charts from the recorded timings.
