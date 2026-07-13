import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table

REMOVE_OUTLIERS = True
OUTLIER_METHOD = "iqr"   # "iqr" or "percentile"
IQR_MULTIPLIER = 2     # standard IQR fence multiplier; increase to be less aggressive
PERCENTILE_RANGE = (1, 99)  # used only if OUTLIER_METHOD == "percentile"

name = "spag"
input_fits = name+".fits"

radigits = 30
decdigits = 30
#
minRA = 5.490024024#spaghetti
maxRA = 5.820754755
minDEC = 25.534534535
maxDEC = 30.441441441

# minRA = 5.22834
# maxRA = 5.777
# minDEC = 19.84
# maxDEC = 24.94

resRA = 1000
resDEC = 1000

beamwidthRA =  0.66537725863748/15  # hours (full width)
beamwidthDEC =  0.66537725863748   # degrees (full width)

t = Table.read(input_fits)

ra_deg = np.asarray(t["CRVAL2"], dtype=float)
dec_deg = np.asarray(t["CRVAL3"], dtype=float)
ra_hours = ra_deg / 15.0

data = np.asarray(t["DATA"], dtype=float)   # shape (N, 1024)
power_per_row = data[:, 1:].mean(axis=1)     # excludes bad channel 0

mask = (
    (ra_hours >= minRA) & (ra_hours <= maxRA) &
    (dec_deg >= minDEC) & (dec_deg <= maxDEC)
)

source_ra = ra_hours[mask]
source_dec = dec_deg[mask]
power = power_per_row[mask]

grid_ra = np.linspace(maxRA, minRA, resRA)
grid_dec = np.linspace(maxDEC, minDEC, resDEC)

ra_half = beamwidthRA / 2
dec_half = beamwidthDEC / 2
epsilon = 1e-6

ra_step = (grid_ra[-1] - grid_ra[0]) / (resRA - 1)
dec_step = (grid_dec[-1] - grid_dec[0]) / (resDEC - 1)

col_radius = int(np.ceil(ra_half / abs(ra_step))) + 1
row_radius = int(np.ceil(dec_half / abs(dec_step))) + 1

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
        continue

    sub_ra = grid_ra[col_lo:col_hi]
    sub_dec = grid_dec[row_lo:row_hi]

    delta_ra = sub_ra[None, :] - source_ra[i]
    delta_dec = sub_dec[:, None] - source_dec[i]

    norm_dist = np.sqrt((delta_ra / ra_half) ** 2 + (delta_dec / dec_half) ** 2)
    within_beam = norm_dist <= 1.0

    weight = np.where(within_beam, 1.0 / (norm_dist ** 2 + epsilon), 0.0)

    power_sum[row_lo:row_hi, col_lo:col_hi] += weight * power[i]
    weight_sum[row_lo:row_hi, col_lo:col_hi] += weight

    if (i + 1) % 500 == 0 or i == n - 1:
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

table.to_csv("fitsv2" + name + str(resRA) + "x" + str(resDEC) + ".csv")

total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print(f"done in {total_time}")

# --- toggle interpolation on/off ---

# compute color scale from REAL data only, before any gap-filling

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
ax.set_title(name+" Beam Map")

if ra_values[0] > ra_values[-1]:
    ax.invert_xaxis()

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Power (dBFS)")

plt.tight_layout()
plt.savefig("beam_visualization.png", dpi=150)
plt.show()