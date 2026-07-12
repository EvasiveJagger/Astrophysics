import math
import os
import sys
import astropy
import numpy as np
import pandas as pd
name=""
radigits = 6
decdigits = 6
#Gas mask RA: 3.98, 4.13 DEC: 35.7,36.7
# name="spag"
# df = pd.read_csv("cleanedspagdata.csv")
# minRA = 5.466 #spaghetti
# maxRA = 5.866
# minDEC = 25
# maxDEC = 31

name="krabby"
df = pd.read_csv("krabby.csv")
minRA = 5.1313#spaghetti
maxRA = 5.777
minDEC = 19.40
maxDEC = 24.94
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

resRA = 100
resDEC = 100

beamwidthRA = 0.028   # hours (full width)
beamwidthDEC = 0.42   # degrees (full width)

# Read the data




df = df[
    df["True RA"].between(minRA, maxRA) &
    df["True DEC"].between(minDEC, maxDEC)
]  # spag bounds

source_ra = df["True RA"].to_numpy(dtype=float)     # (N,) hours
source_dec = df["True DEC"].to_numpy(dtype=float)   # (N,) degrees
power = df["Power (dBFS)"].to_numpy(dtype=float)    # (N,)

# Evenly spaced grid axes, built purely from bounds + resolution
grid_ra = np.linspace(maxRA, minRA, resRA)                    # now descending
grid_dec = np.linspace(maxDEC, minDEC, resDEC)

ra_half = beamwidthRA / 2    # hours
dec_half = beamwidthDEC / 2  # degrees

print("calculating beam")
power_sum = np.zeros((len(grid_dec), len(grid_ra)))
weight_sum = np.zeros((len(grid_dec), len(grid_ra)))
z=0
# Small floor to avoid divide-by-zero when a source sits exactly on a grid point
epsilon = 1e-6

for i in range(len(grid_dec)):
    z += 1
    print(str(z) + "/" + str(len(grid_dec)))
    delta_dec = grid_dec[i] - source_dec                        # (N,) degrees
    delta_ra = grid_ra[:, None] - source_ra[None, :]             # (C, N) hours

    # Elliptical normalized distance (1.0 = at the beam edge)
    norm_dist = np.sqrt(
        (delta_ra / ra_half) ** 2
        + (delta_dec[None, :] / dec_half) ** 2
    )

    within_beam = norm_dist <= 1.0                               # (C, N)

    # Inverse-square weighting by normalized distance from beam center
    weight = 1.0 / (norm_dist ** 2 + epsilon)
    weight = np.where(within_beam, weight, 0.0)                  # zero out points outside beam

    power_sum[i] = (weight * power[None, :]).sum(axis=1)
    weight_sum[i] = weight.sum(axis=1)

with np.errstate(invalid="ignore"):
    beam_values = power_sum / weight_sum

table = pd.DataFrame(
    beam_values,
    index=np.round(grid_dec, decdigits),
    columns=np.round(grid_ra, radigits),
)


# Save
table.to_csv("v2"+ name + str(resRA) + "x" + str(resDEC) + ".csv")

