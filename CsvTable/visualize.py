import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- toggle interpolation on/off ---

# compute color scale from REAL data only, before any gap-filling
REMOVE_OUTLIERS = True
OUTLIER_METHOD = "iqr"   # "iqr" or "percentile"
IQR_MULTIPLIER = 6      # standard IQR fence multiplier; increase to be less aggressive
PERCENTILE_RANGE = (1, 9)  # used only if OUTLIER_METHOD == "percentile"

table = pd.read_csv("v2krabby500x500.csv", index_col=0)

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
ax.set_title("Spaghetti Beam Map")

if ra_values[0] > ra_values[-1]:
    ax.invert_xaxis()

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Power (dBFS)")

plt.tight_layout()
plt.savefig("beam_visualization.png", dpi=150)
plt.show()