import math
import os
import sys
import astropy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
name=""
radigits = 30
decdigits = 30
# --- toggle interpolation on/off ---

# compute color scale from REAL data only, before any gap-filling
REMOVE_OUTLIERS = True
OUTLIER_METHOD = "iqr"   # "iqr" or "percentile"
IQR_MULTIPLIER = 4      # standard IQR fence multiplier; increase to be less aggressive
PERCENTILE_RANGE = (1, 9)  # used only if OUTLIER_METHOD == "percentile"

#Gas mask RA: 3.98, 4.13 DEC: 35.7,36.7
name="spag"
df = pd.read_csv("cleanedspagdata.csv")
minRA = 5.490024024 #spaghetti
maxRA = 5.820754755
minDEC = 25.534534535
maxDEC = 30.441441441

# name="krabby"
# df = pd.read_csv("krabby.csv")
# minRA = 5.22834#spaghetti
# maxRA = 5.777
# minDEC = 19.84
# maxDEC = 24.94
#name="spacesquir"
# df = pd.read_csv("spaces.csv")
# minRA = 0.58333 #spaceballs
# maxRA = 1.404882
# minDEC = 54.20389
# maxDEC = 64.120498

#name="RRW"
#df = pd.read_csv("RRW.csv")
# minRA = 0.606061 #rrw
# maxRA = 1.378788
# minDEC = 54.69697
# maxDEC = 63.484848

# name="gasmask"
# df = pd.read_csv("gasmask.csv")
# minRA = 3.869 #Gas mask
# maxRA = 4.21
# minDEC = 35.458
# maxDEC = 36.8823

resRA = 75
resDEC = 75

#0.42
beamwidth = 0.56
beamwidthRA = beamwidth/15   # hours (full width)
beamwidthDEC = beamwidth   # degrees (full width)

# Read the data
start_time = time.time()


df = df[
    df["True RA"].between(minRA, maxRA) &
    df["True DEC"].between(minDEC, maxDEC)
]

source_ra = df["True RA"].to_numpy(dtype=float)
source_dec = df["True DEC"].to_numpy(dtype=float)
power = df["Power (dBFS)"].to_numpy(dtype=float)

grid_ra = np.linspace(maxRA, minRA, resRA)     # descending
grid_dec = np.linspace(maxDEC, minDEC, resDEC)  # descending

ra_half = beamwidthRA / 2
dec_half = beamwidthDEC / 2
epsilon = 1e-6

# grid spacing (constant, since linspace) — used to convert a source's
# RA/Dec into fractional grid index, and beam half-width into a cell radius
ra_step = (grid_ra[-1] - grid_ra[0]) / (resRA - 1)     # negative (descending)
dec_step = (grid_dec[-1] - grid_dec[0]) / (resDEC - 1)  # negative (descending)

col_radius = int(np.ceil(ra_half / abs(ra_step))) + 1
row_radius = int(np.ceil(dec_half / abs(dec_step))) + 1

# fractional index of each source point within the grid
col_center = (source_ra - grid_ra[0]) / ra_step
row_center = (source_dec - grid_dec[0]) / dec_step

power_sum = np.zeros((resDEC, resRA))
weight_sum = np.zeros((resDEC, resRA))

print("calculating beam")
n = len(source_ra)
start_time = time.time()

for i in range(n):
    col_c = col_center[i]
    row_c = row_center[i]

    col_lo = max(0, int(np.floor(col_c - col_radius)))
    col_hi = min(resRA, int(np.ceil(col_c + col_radius)) + 1)
    row_lo = max(0, int(np.floor(row_c - row_radius)))
    row_hi = min(resDEC, int(np.ceil(row_c + row_radius)) + 1)

    if col_lo >= col_hi or row_lo >= row_hi:
        continue  # source falls entirely outside the grid

    sub_ra = grid_ra[col_lo:col_hi]     # (w,)
    sub_dec = grid_dec[row_lo:row_hi]   # (h,)

    delta_ra = sub_ra[None, :] - source_ra[i]     # (1, w)
    delta_dec = sub_dec[:, None] - source_dec[i]  # (h, 1)

    norm_dist = np.sqrt((delta_ra / ra_half) ** 2 + (delta_dec / dec_half) ** 2)  # (h, w)
    within_beam = norm_dist <= 1.0

    weight = np.where(within_beam, 1.0 / (norm_dist ** 2 + epsilon), 0.0)

    power_sum[row_lo:row_hi, col_lo:col_hi] += weight * power[i]
    weight_sum[row_lo:row_hi, col_lo:col_hi] += weight

    if (i + 1) % 100 == 0 or i == n - 1:
        elapsed = time.time() - start_time
        avg = elapsed / (i + 1)
        remaining = avg * (n - (i + 1))
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"{i+1}/{n}  elapsed {elapsed_str}  ETA {eta_str}")

with np.errstate(invalid="ignore"):
    beam_values = power_sum / weight_sum

table = pd.DataFrame(
    beam_values,
    index=np.round(grid_dec, decdigits),
    columns=np.round(grid_ra, radigits),
)

table.to_csv("v2" + name + str(resRA) + "x" + str(resDEC) + ".csv")

total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print(f"done in {total_time}")

dec_values = table.index.to_numpy(dtype=float)
ra_values = table.columns.to_numpy(dtype=float)
power = table.to_numpy(dtype=float)

if REMOVE_OUTLIERS:
    valid = power[~np.isnan(power)]

    if OUTLIER_METHOD == "iqr":
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        lower_fence = q1 - IQR_MULTIPLIER * iqr
        upper_fence = q3 + IQR_MULTIPLIER * iqr
    else:  # "percentile"
        lower_fence, upper_fence = np.percentile(valid, PERCENTILE_RANGE)

    print(f"outlier fences: [{lower_fence:.3f}, {upper_fence:.3f}]  "
          f"(removing {np.sum((valid < lower_fence) | (valid > upper_fence))} of {len(valid)} points)")

    # turn out-of-range values into NaN so they're excluded like other gaps
    power = np.where((power < lower_fence) | (power > upper_fence), np.nan, power)

# compute color scale from REAL (post-outlier-removal) data only, before any gap-filling
vmin = np.nanmin(power)
vmax = np.nanmax(power)



dec_values = table.index.to_numpy(dtype=float)
ra_values = table.columns.to_numpy(dtype=float)*15
power = table.to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(8, 7))

mesh = ax.pcolormesh(
    ra_values,
    dec_values,
    power,
    shading="nearest",
    cmap="jet",
    vmin=vmin,
    vmax=vmax,
)

ax.set_xlabel("Right Ascension (degrees)")
ax.set_ylabel("Declination (degrees)")
ax.set_title(name+ " Beam Map")

if ra_values[0] > ra_values[-1]:
    ax.invert_xaxis()

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Power (dBFS)")

plt.tight_layout()
plt.savefig("beam_visualization.png", dpi=150)
plt.show()