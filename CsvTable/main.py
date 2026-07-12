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


# Create empty telescope map

# Count hits for averaging
hits = pd.DataFrame(
    0,
    index=table.index,
    columns=table.columns
)
num=0
print("calculating beam")
for _, source in df.iterrows():
    num+=1
    print(str(num)+"/"+str(len(df)))
    source_ra = source["True RA"]       # hours
    source_dec = source["True DEC"]     # degrees
    power = source["Power (dBFS)"]


    for dec in table.index:
        for ra in table.columns:

            delta_ra = (ra - source_ra) * 15
            delta_dec = dec - source_dec

            distance = math.sqrt((delta_ra * math.cos(math.radians(source_dec)))**2+delta_dec**2)

            if distance <= beam_radius:

                table.loc[dec, ra] += power
                hits.loc[dec, ra] += 1



# Average overlapping beams
table = table / hits.replace(0, math.nan)

# Empty cells = 0
table = table.fillna(0)

# Save
table.to_csv("spaoutputbeam"+str(radigits)+"x"+str(decdigits)+".csv")