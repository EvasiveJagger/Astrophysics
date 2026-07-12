# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import math
#
# def generate_ra_columns(file, digits,asc):
#     cols = []
#     #range(len(file['RAFix']))
#     for x in range(10):
#         RA = round(file['True RA'][x],digits)
#         if RA not in cols:
#             if len(cols)==0:
#                 cols.append(RA)
#             else:
#                 if asc:
#                     z = 0
#                     while z<len(cols):
#                         if RA==cols[z]:
#                             break
#                         if RA>cols[z]:
#                             z+=1
#                             if z==len(cols):
#                                 cols.append(RA)
#                         elif RA<cols[z]:
#                             cols.insert(z, RA)
#                             break
#                 else:
#                     z = 0
#                     while z < len(cols):
#                         if RA == cols[z]:
#                             break
#                         if RA > cols[z]:
#                             cols.insert(z, RA)
#                             break
#                         elif RA < cols[z]:
#                             z += 1
#                             if z == len(cols):
#                                 cols.append(RA)
#         #print("list: "+str(cols))
#     return cols
#
# def generate_dec_rows(file,digits, asc):
#     rows = []
#     # range(len(file['RAFix']))
#     for x in range(10):
#         Dec = round(file['True DEC'][x], digits)
#         if Dec not in rows:
#             if len(rows) == 0:
#                 rows.append(Dec)
#             else:
#                 if asc:
#                     z = 0
#                     while z < len(rows):
#                         if Dec == rows[z]:
#                             break
#                         if Dec > rows[z]:
#                             z += 1
#                             if z == len(rows):
#                                 rows.append(Dec)
#                         elif Dec < rows[z]:
#                             rows.insert(z, Dec)
#                             break
#                 else:
#                     z = 0
#                     while z < len(rows):
#                         if Dec == rows[z]:
#                             break
#                         if Dec > rows[z]:
#                             rows.insert(z, Dec)
#                             break
#                         elif Dec < rows[z]:
#                             z += 1
#                             if z == len(rows):
#                                 rows.append(Dec)
#         # print("list: "+str(cols))
#     return rows
# def generatetable(columns,rows,file,digits):
#     datatable = np.zeros((len(columns),len(rows)));
#
#     for x in range(10):
#         power = file["Power (dBFS)"][x]
#         ra = round(file["True RA"][x],digits)
#         dec = round(file["True DEC"][x],digits)
#         if ra in columns and dec in rows:
#             if datatable[columns.index(ra),rows.index(dec)]==0:
#                 datatable[columns.index(ra), rows.index(dec)] = power;
#             else:
#                 datatable[columns.index(ra), rows.index(dec)] = (datatable[columns.index(ra), rows.index(dec)]+power)/2
#     return datatable
# def generatedataframe(table,columns,rows):
#     output = pd.DataFrame(
#         {
#             "Dec": [rows]
#         }
#     )
#     for x in range(len(columns)):
#         output[columns[x]] = table[x].tolist()
#     return output


#
# df = pd.read_csv("cleanedspagdata.csv")
# co = generate_ra_columns(df,5,True)
# ro = generate_dec_rows(df,5,False)
# table = generatetable(co,ro,df,5)
# out = generatedataframe(table,co,ro)
# out.to_csv("output")


import math
import os
import sys

import numpy as np
import pandas as pd

radigits = 4
decdigits = 3

beamwidthRA = 0.028
beamwidthDEC = 0.42
# Read the data
df = pd.read_csv("cleanedspagdata.csv")

# Round coordinates
df["RA"] = df["True RA"].round(radigits)
df["Dec"] = df["True DEC"].round(decdigits)
ra_min = 5.4
dec_min = -20

# df = df[
#     df["True RA"].between(3.98,4.13) &
#     df["True DEC"].between(35.7,36.7)
# ] # gas mask bounds
df = df[
    df["True RA"].between(5.4,5.8) &
    df["True DEC"].between(25.7,30.2)
] #spag bounds
# Create table
table = df.pivot_table(
    index="Dec",
    columns="RA",
    values="Power (dBFS)",
    aggfunc="mean",     # average duplicates
    fill_value=0        # empty cells become 0
)

# Optional sorting
table = table.sort_index(ascending=False)          # Dec descending
table = table.reindex(sorted(table.columns), axis=1)  # RA ascending

# for dec, row in table.iterrows():
#     for ra, power in row.items():
#         if power == 0:
#             continue
#         new_df = df[
#             df["True RA"].between(ra-beamwidthRA, ra+beamwidthRA) &
#             df["True DEC"].between(dec-beamwidthDEC, dec+beamwidthDEC)
#         ]

table.to_csv("beam.csv")

beam_radius = 0.42 / 2   # degrees

# Fresh accumulators for the beam convolution.
# IMPORTANT: these must start at zero, not reuse the pivoted `table` values above
# (that table already contains raw per-sample averages, and adding beam
# contributions on top of it double-counted every cell that happened to sit
# on an actual sample point -- which is dense right along the scan path.
# That was the source of the bright streak tracking the middle of the scan).
grid_dec = table.index.to_numpy(dtype=float)      # (R,)
grid_ra = table.columns.to_numpy(dtype=float)     # (C,)

source_ra = df["True RA"].to_numpy(dtype=float)     # (N,) hours
source_dec = df["True DEC"].to_numpy(dtype=float)   # (N,) degrees
power = df["Power (dBFS)"].to_numpy(dtype=float)    # (N,)

print("calculating beam")

power_sum = np.zeros((len(grid_dec), len(grid_ra)))
hit_count = np.zeros((len(grid_dec), len(grid_ra)))

# Vectorized beam convolution: broadcast every grid cell against every
# source sample at once instead of a Python triple-nested loop.
for i in range(len(grid_dec)):
    delta_dec = grid_dec[i] - source_dec                      # (N,)
    delta_ra = (grid_ra[:, None] - source_ra[None, :]) * 15    # (C, N)
    distance = np.sqrt((delta_ra * np.cos(np.radians(source_dec))[None, :]) ** 2 + delta_dec[None, :] ** 2)

    within_beam = distance <= beam_radius                      # (C, N)
    power_sum[i] = within_beam @ power
    hit_count[i] = within_beam.sum(axis=1)

with np.errstate(invalid="ignore"):
    beam_values = power_sum / hit_count

table = pd.DataFrame(beam_values, index=table.index, columns=table.columns)

# Empty cells = 0
table = table.fillna(0)

# Save
table.to_csv("spaoutputbeam"+str(radigits)+"x"+str(decdigits)+".csv")