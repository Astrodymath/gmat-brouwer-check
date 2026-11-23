# Checks for GMAT's BrouwerMeanLong conversion near the critical inclination

Scripts for the report "BrouwerMeanLong near the critical inclinations", in two forms:

- three GMAT scripts in `scripts/`, which show the three symptoms in GMAT itself and need nothing else;
- two Python scripts, which call GMAT R2026a through its Python API (`StateConversionUtil`) for the full
  numbers of the report (GMAT R2026a ships the API for Python 3.9 to 3.14);
- `fixed_values.py`, which needs no GMAT and prints what the three GMAT scripts give with the proposed fix.

Tested on Linux with gmat-ubuntu-x64-R2026a (GmatConsole, and the API with Python 3.13). The three GMAT
scripts were also run on Windows with gmat-win-R2026a (GmatConsole), with the same output to the last digit.

## GMAT scripts

From GMAT's `bin` folder (or open them in the GMAT GUI and run), on Linux:

```
./GmatConsole <this folder>/scripts/BL_critical_63.script    # runs; prints AOP 270.008 (Short) and 329.966 (Long), and 10148 km
./GmatConsole <this folder>/scripts/BL_critical_116.script   # stops: "The input ... to ASin() is not within -1.0 and 1.0"
./GmatConsole <this folder>/scripts/BL_inverse_63.script     # Short works; Long stops: "... only if mean ECC is smaller than 0.99"
```

On Windows, in PowerShell:

```
.\GmatConsole.exe <this folder>\scripts\BL_critical_63.script
.\GmatConsole.exe <this folder>\scripts\BL_critical_116.script
.\GmatConsole.exe <this folder>\scripts\BL_inverse_63.script
```

| script | what it does |
|---|---|
| `BL_critical_63.script` | the same mean elements (26600 km, 0.74, 63.4°, RAAN 20°, AOP 270°, MA 10°) as BrouwerMeanShort and as BrouwerMeanLong; reports both osculating states and the distance between them |
| `BL_critical_116.script` | the same at i = 116.5°, AOP 100°, MA 50°: BrouwerMeanLong throws while the script is read |
| `BL_inverse_63.script` | an osculating Molniya state (Keplerian, i = 63.4°) converted to BrouwerMeanShort (works) and to BrouwerMeanLong (throws) |

`expected/scripts/*.console.txt` keep only the relevant console lines of the Linux run, in their order; GMAT
also prints its start-up and kernel-loading lines between them. In `BL_critical_63` the warning appears while the
script is read, before "Successfully interpreted the script"; in `BL_inverse_63` the warnings appear during the
run. The report files (`*.txt`) are written to GMAT's `output` folder.

## Requirements

- For the two API scripts: GMAT R2026a (any platform), Python 3.9 to 3.14 matching one of the folders in
  `<GMAT>/bin/gmatpy/`. `fixed_values.py` needs no GMAT and runs with any recent Python 3.
- numpy; scipy (imported by `truth.py`, which `bl_numpy.py` uses for its constants).

## Python scripts

From this folder, with `<GMAT>` the GMAT installation folder:

```
python gmat_check.py <GMAT>/bin > gmat_check.out
python real_vs_transcription.py <GMAT>/bin > real_vs_transcription.out
```

Compare with `expected/`. The last digits of the round-off-level differences (1e-7 m) may change from
platform to platform; nothing else should.

## Values with the proposed fix (no GMAT needed)

```
python fixed_values.py > fixed_values.out
```

For the three GMAT scripts, it prints GMAT's results (from the transcription `gmat_bl.py`), the results with
the proposed fix (`gmat_bl.long(..., t2=True)`: l+g+h and the short-period terms kept, 1/(1−5θ²) replaced by
Phipps' T2, and `b7` written in Brouwer's form, see below), and the same with the independent implementation
`bl_numpy.py`. It then integrates the J2–J5 equations of motion for one day from both osculating states, and
compares the fix with `bl_numpy.py` and with GMAT as it is on a grid of inclinations (about 15 s).

`b7` is written with (1−15θ²)/(1−5θ²) and (1−7θ²)/(1−5θ²). These equal Brouwer's
(1−11θ²−40θ⁴/(1−5θ²))/(1−θ²) and (1−3θ²−8θ⁴/(1−5θ²))/(1−θ²), the forms used in `a1` and `a2`, only with the
raw divisor. With the smoothed divisor, the fix uses Brouwer's forms, so that `b7` = `b13`/sin i and the
long-period δI still follows from δe (conservation of H); a plain substitution in `b7` moves the osculating i
by 0.001° in the Molniya example (53 m in position).

## Files

| file | content |
|---|---|
| `gmat_check.py` | GMAT itself: (1) one Molniya example, Short vs Long; (2) Long − Short position difference at the same mean elements, 200 random angles, with GMAT's `bisubc`; (3) scan of the inclination in steps of 0.01° near 63.4° and 116.6°: where Long differs from Short by more than 1000 km and where Long throws; (4) `CartesianToBrouwerMeanLong` inside and outside the band |
| `real_vs_transcription.py` | GMAT itself vs the line-by-line transcription `gmat_bl.py`, and vs the first-order Brouwer–Lyddane map of `bl_numpy.py` (raw critical divisor, GMAT's constants), 13 orbits × 100 random angles |
| `gmat_bl.py` | line-by-line Python transcription of `BrouwerMeanShortToOsculatingElements` and `BrouwerMeanLongToOsculatingElements` (R2026a); `crit_branch=False` removes the critical-inclination branch; `t2=True` applies the proposed fix (not in GMAT); with the default arguments the results are bit for bit those of v01-02 |
| `fixed_values.py` | the three GMAT script cases with the proposed fix, an independent check, a one-day numerical integration, and a grid comparison; needs only numpy and scipy |
| `bl_numpy.py`, `truth.py` | an independent first-order Brouwer–Lyddane implementation (written to check Orekit's, with its five coefficient differences corrected) and its constants; used here as the reference theory and for Keplerian → Cartesian |
| `expected/` | the outputs of the three Python scripts, and of the three GMAT scripts in `expected/scripts/` |
