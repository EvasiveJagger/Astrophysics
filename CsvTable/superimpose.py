"""
Superimpose the CSV beam map and the FITS beam map as a single bivariate image.

  SATURATION  <- one dataset
  BRIGHTNESS  <- the other dataset
  HUE         <- fixed, set by BASE_COLOR

Which dataset drives which channel is controlled by SWAP_CHANNELS, and each
channel's direction can be flipped independently with INVERT_SAT / INVERT_VALUE.

Both grids are resampled onto a common RA/Dec grid first, then each channel is
independently normalized and quantized, so the image uses exactly
N_SAT x N_VAL unique colors.
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

# which dataset drives which channel
SWAP_CHANNELS = False    # False: CSV -> saturation, FITS -> brightness
                         # True:  FITS -> saturation, CSV -> brightness

# which end of each dataset's power range maps to which end of its channel.
# these apply to the CHANNEL, so they mean the same thing regardless of SWAP_CHANNELS.
INVERT_SAT = False       # False: high power -> saturated.  True: high power -> washed out
INVERT_VALUE = False     # False: high power -> bright.     True: high power -> dark

N_SAT = 10   # number of saturation steps
N_VAL = 10   # number of brightness steps

# floors keep the dimmest/least-saturated bin from collapsing to black / gray
SAT_RANGE = (0.15, 1.00)
VAL_RANGE = (0.25, 1.00)

# percentile clip for normalization -- resists the outliers we found earlier
CLIP_PCT = (2, 98)

RES = 50          # resolution of the common grid
RA_IN_HOURS = True # set False if your grid CSVs already store RA in degrees

# ---------------------------------------------------------------- helpers


def load_grid(path, ra_in_hours=RA_IN_HOURS):
    """Read one gridded CSV -> (dec_asc, ra_deg_asc, values) with ascending axes."""
    t = pd.read_csv(path, index_col=0)
    dec = t.index.to_numpy(dtype=float)
    ra = t.columns.to_numpy(dtype=float)
    if ra_in_hours:
        ra = ra * 15.0
    v = t.to_numpy(dtype=float)

    # RegularGridInterpolator requires strictly ascending axes; the grids are
    # written descending, so flip whichever axis needs it
    if dec[0] > dec[-1]:
        dec, v = dec[::-1], v[::-1, :]
    if ra[0] > ra[-1]:
        ra, v = ra[::-1], v[:, ::-1]
    return dec, ra, v


def resample(dec, ra, v, dec_new, ra_new):
    """Bilinear resample onto a new grid; outside coverage -> NaN."""
    f = RegularGridInterpolator(
        (dec, ra), v, bounds_error=False, fill_value=np.nan, method="linear"
    )
    DD, RR = np.meshgrid(dec_new, ra_new, indexing="ij")
    return f(np.stack([DD.ravel(), RR.ravel()], axis=-1)).reshape(DD.shape)


def normalize_quantize(v, n_levels, out_range, invert=False, clip_pct=CLIP_PCT):
    """
    Robustly scale to 0..1, optionally invert, snap to n_levels discrete steps,
    then map into out_range. Returns (channel, low_power_value, high_power_value)
    where the last two are the raw data values at the ends of the channel.
    NaNs stay NaN.
    """
    good = v[np.isfinite(v)]
    lo, hi = np.percentile(good, clip_pct)
    if hi <= lo:
        hi = lo + 1e-12

    x = np.clip((v - lo) / (hi - lo), 0.0, 1.0)

    # inverting here (before quantizing) keeps the bins evenly spaced
    if invert:
        x = 1.0 - x

    idx = np.clip((x * n_levels).astype(float), 0, n_levels - 1e-9)
    idx = np.floor(np.where(np.isfinite(idx), idx, np.nan))
    level = idx / (n_levels - 1) if n_levels > 1 else np.zeros_like(idx)

    out = out_range[0] + level * (out_range[1] - out_range[0])

    # what raw power value sits at the bottom vs the top of this channel
    low_end, high_end = (hi, lo) if invert else (lo, hi)
    return out, low_end, high_end


# ---------------------------------------------------------------- build

for p in (CSV_GRID, FITS_GRID):
    print(f"{p}  modified {time.ctime(os.path.getmtime(p))}")

dec_c, ra_c, v_csv = load_grid(CSV_GRID)
dec_f, ra_f, v_fits = load_grid(FITS_GRID)

# common grid = the overlap of the two footprints
dec_lo, dec_hi = max(dec_c[0], dec_f[0]), min(dec_c[-1], dec_f[-1])
ra_lo, ra_hi = max(ra_c[0], ra_f[0]), min(ra_c[-1], ra_f[-1])
if dec_lo >= dec_hi or ra_lo >= ra_hi:
    raise ValueError("the two grids do not overlap on the sky")

dec_g = np.linspace(dec_lo, dec_hi, RES)
ra_g = np.linspace(ra_lo, ra_hi, RES)

csv_on_grid = resample(dec_c, ra_c, v_csv, dec_g, ra_g)
fits_on_grid = resample(dec_f, ra_f, v_fits, dec_g, ra_g)

# decide which dataset drives which channel
if SWAP_CHANNELS:
    sat_src, sat_label = fits_on_grid, "FITS"
    val_src, val_label = csv_on_grid, "CSV"
else:
    sat_src, sat_label = csv_on_grid, "CSV"
    val_src, val_label = fits_on_grid, "FITS"

sat, sat_low, sat_high = normalize_quantize(sat_src, N_SAT, SAT_RANGE, invert=INVERT_SAT)
val, val_low, val_high = normalize_quantize(val_src, N_VAL, VAL_RANGE, invert=INVERT_VALUE)

hue = rgb_to_hsv(np.array(to_rgb(BASE_COLOR)))[0]

missing = ~np.isfinite(sat) | ~np.isfinite(val)
hsv = np.dstack([
    np.full_like(sat, hue),
    np.nan_to_num(sat),
    np.nan_to_num(val),
])
rgb = hsv_to_rgb(hsv)
rgb[missing] = 0.85  # light gray where either map has no coverage

# ---------------------------------------------------------------- plot

fig, (ax, lax) = plt.subplots(
    1, 2, figsize=(11, 7), gridspec_kw={"width_ratios": [4, 1]}
)

ax.imshow(
    rgb,
    origin="lower",
    extent=[ra_g[0], ra_g[-1], dec_g[0], dec_g[-1]],
    aspect="auto",
    interpolation="nearest",
)
ax.invert_xaxis()  # RA increases right-to-left
ax.set_xlabel("Right Ascension (degrees)")
ax.set_ylabel("Declination (degrees)")
ax.set_title(
    f"{sat_label} (saturation) x {val_label} (brightness) -- {N_SAT}x{N_VAL} colors"
)

# 2D legend: every color actually used in the image
s_lv = np.linspace(SAT_RANGE[0], SAT_RANGE[1], N_SAT)
v_lv = np.linspace(VAL_RANGE[0], VAL_RANGE[1], N_VAL)
SS, VV = np.meshgrid(s_lv, v_lv)
key = hsv_to_rgb(np.dstack([np.full_like(SS, hue), SS, VV]))

lax.imshow(key, origin="lower", extent=[0, N_SAT, 0, N_VAL], aspect="auto",
           interpolation="nearest")

# tick the corners with raw power values so the direction is unambiguous
lax.set_xticks([0, N_SAT])
lax.set_xticklabels([f"{sat_low:.2f}", f"{sat_high:.2f}"], fontsize=8)
lax.set_yticks([0, N_VAL])
lax.set_yticklabels([f"{val_low:.2f}", f"{val_high:.2f}"], fontsize=8)
lax.set_xlabel(f"{sat_label} power\n(saturation)")
lax.set_ylabel(f"{val_label} power\n(brightness)")
lax.set_title("color key")

plt.tight_layout()
plt.savefig("bivariate_overlay.png", dpi=150)
plt.show()