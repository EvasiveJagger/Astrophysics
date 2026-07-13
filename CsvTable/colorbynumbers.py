"""
Color-by-numbers version of the bivariate CSV x FITS overlay.

  The KEY (right panel) keeps the colors, with a number printed on each swatch.
  The MAP (left panel) has no fill -- each cell just shows the number of the
  color it should be painted.

Color number = val_bin * N_SAT + sat_bin + 1, running 1 .. N_SAT*N_VAL.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------- settings

CSV_GRID = "v2spag75x75.csv"
FITS_GRID = "fitsv2spag75x75.csv"

BASE_COLOR = "#ff0000"   # any matplotlib color; only its hue is used

SWAP_CHANNELS = False    # False: CSV -> saturation, FITS -> brightness
                         # True:  FITS -> saturation, CSV -> brightness

INVERT_SAT = False       # False: high power -> saturated.  True: high power -> washed out
INVERT_VALUE = False     # False: high power -> bright.     True: high power -> dark

N_SAT = 10   # number of saturation steps
N_VAL = 10   # number of brightness steps

SAT_RANGE = (0.15, 1.00)
VAL_RANGE = (0.25, 1.00)

CLIP_PCT = (2, 98)

# color-by-numbers only works at low resolution -- every cell gets a printed
# number, so RES=40 means 1600 numbers on the page. Raise FIG_SIZE/DPI before RES.
RES = 50
CELL_FONTSIZE = 9      # font size of the numbers in the map
KEY_FONTSIZE = 7         # font size of the numbers on the color key
FIG_SIZE = (16, 11)
DPI = 300
SHOW_GRIDLINES = True

RA_IN_HOURS = True

# ---------------------------------------------------------------- helpers


def load_grid(path, ra_in_hours=RA_IN_HOURS):
    t = pd.read_csv(path, index_col=0)
    dec = t.index.to_numpy(dtype=float)
    ra = t.columns.to_numpy(dtype=float)
    if ra_in_hours:
        ra = ra * 15.0
    v = t.to_numpy(dtype=float)
    if dec[0] > dec[-1]:
        dec, v = dec[::-1], v[::-1, :]
    if ra[0] > ra[-1]:
        ra, v = ra[::-1], v[:, ::-1]
    return dec, ra, v


def resample(dec, ra, v, dec_new, ra_new):
    f = RegularGridInterpolator(
        (dec, ra), v, bounds_error=False, fill_value=np.nan, method="linear"
    )
    DD, RR = np.meshgrid(dec_new, ra_new, indexing="ij")
    return f(np.stack([DD.ravel(), RR.ravel()], axis=-1)).reshape(DD.shape)


def quantize(v, n_levels, invert=False, clip_pct=CLIP_PCT):
    """Return (bin_index float array with NaNs preserved, low_end, high_end)."""
    good = v[np.isfinite(v)]
    lo, hi = np.percentile(good, clip_pct)
    if hi <= lo:
        hi = lo + 1e-12

    x = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        x = 1.0 - x

    idx = np.clip(x * n_levels, 0, n_levels - 1e-9)
    idx = np.floor(np.where(np.isfinite(x), idx, np.nan))

    low_end, high_end = (hi, lo) if invert else (lo, hi)
    return idx, low_end, high_end


def level_to_channel(idx, n_levels, out_range):
    lvl = idx / (n_levels - 1) if n_levels > 1 else np.zeros_like(idx)
    return out_range[0] + lvl * (out_range[1] - out_range[0])


# ---------------------------------------------------------------- build

for p in (CSV_GRID, FITS_GRID):
    print(f"{p}  modified {time.ctime(os.path.getmtime(p))}")

dec_c, ra_c, v_csv = load_grid(CSV_GRID)
dec_f, ra_f, v_fits = load_grid(FITS_GRID)

dec_lo, dec_hi = max(dec_c[0], dec_f[0]), min(dec_c[-1], dec_f[-1])
ra_lo, ra_hi = max(ra_c[0], ra_f[0]), min(ra_c[-1], ra_f[-1])
if dec_lo >= dec_hi or ra_lo >= ra_hi:
    raise ValueError("the two grids do not overlap on the sky")

dec_g = np.linspace(dec_lo, dec_hi, RES)
ra_g = np.linspace(ra_lo, ra_hi, RES)

csv_on_grid = resample(dec_c, ra_c, v_csv, dec_g, ra_g)
fits_on_grid = resample(dec_f, ra_f, v_fits, dec_g, ra_g)

if SWAP_CHANNELS:
    sat_src, sat_label = fits_on_grid, "FITS"
    val_src, val_label = csv_on_grid, "CSV"
else:
    sat_src, sat_label = csv_on_grid, "CSV"
    val_src, val_label = fits_on_grid, "FITS"

sat_idx, sat_low, sat_high = quantize(sat_src, N_SAT, invert=INVERT_SAT)
val_idx, val_low, val_high = quantize(val_src, N_VAL, invert=INVERT_VALUE)

# the color number each cell should be painted
color_num = val_idx * N_SAT + sat_idx + 1     # 1 .. N_SAT*N_VAL, NaN where missing

hue = rgb_to_hsv(np.array(to_rgb(BASE_COLOR)))[0]

# ---------------------------------------------------------------- plot

fig, (ax, lax) = plt.subplots(
    1, 2, figsize=FIG_SIZE, gridspec_kw={"width_ratios": [4, 1]}
)

# ---- MAP: no color, just numbers ----
PAD_CELLS = 0.6   # how much to expand the box, in cell widths

dra = (ra_g[-1] - ra_g[0]) / (RES - 1)
dde = (dec_g[-1] - dec_g[0]) / (RES - 1)

ax.set_xlim(ra_g[0] - PAD_CELLS * dra, ra_g[-1] + PAD_CELLS * dra)
ax.set_ylim(dec_g[0] - PAD_CELLS * dde, dec_g[-1] + PAD_CELLS * dde)
ax.invert_xaxis()


if SHOW_GRIDLINES:
    for r in np.append(ra_g, ra_g[-1] + dra) - dra / 2:
        ax.axvline(r, color="0.9", lw=0.2, zorder=0)
    for d in np.append(dec_g, dec_g[-1] + dde) - dde / 2:
        ax.axhline(d, color="0.9", lw=0.2, zorder=0)

for i in range(RES):
    for j in range(RES):
        n = color_num[i, j]
        if not np.isfinite(n):
            continue
        ax.text(
            ra_g[j], dec_g[i], f"{int(n)}",
            ha="center", va="center", fontsize=CELL_FONTSIZE, color="0.15",
        )

ax.set_xlabel("Right Ascension (degrees)")
ax.set_ylabel("Declination (degrees)")
ax.set_title(
    f"color by numbers -- {sat_label} (saturation) x {val_label} (brightness), "
    f"{N_SAT * N_VAL} colors"
)

# ---- KEY: colors, each swatch numbered ----
s_lv = np.arange(N_SAT)
v_lv = np.arange(N_VAL)
SS, VV = np.meshgrid(s_lv, v_lv)
key = hsv_to_rgb(np.dstack([
    np.full(SS.shape, hue),
    level_to_channel(SS.astype(float), N_SAT, SAT_RANGE),
    level_to_channel(VV.astype(float), N_VAL, VAL_RANGE),
]))

lax.imshow(key, origin="lower", extent=[0, N_SAT, 0, N_VAL], aspect="auto",
           interpolation="nearest")

for vi in range(N_VAL):
    for si in range(N_SAT):
        n = vi * N_SAT + si + 1
        # dark text on light swatches, light text on dark ones
        lum = key[vi, si].mean()
        lax.text(
            si + 0.5, vi + 0.5, f"{n}",
            ha="center", va="center", fontsize=KEY_FONTSIZE,
            color="0.1" if lum > 0.55 else "white",
        )

lax.set_xticks([0, N_SAT])
lax.set_xticklabels([f"{sat_low:.2f}", f"{sat_high:.2f}"], fontsize=8)
lax.set_yticks([0, N_VAL])
lax.set_yticklabels([f"{val_low:.2f}", f"{val_high:.2f}"], fontsize=8)
lax.set_xlabel(f"{sat_label} power\n(saturation)")
lax.set_ylabel(f"{val_label} power\n(brightness)")
lax.set_title("color key")

plt.tight_layout()
plt.savefig("color_by_numbers.png", dpi=DPI)
plt.show()