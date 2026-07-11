# This is a sample Python script.
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
latitude = math.radians(35.188056)
longitude = math.radians(-82.875475)
cdate = "2026-07-10"
def get_julian_datetime(date):
    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int((275 * date.month) / 9.0) + date.day + 1721013.5 + (date.hour + date.minute / 60.0 + date.second / math.pow(60,2)) / 24.0 - 0.5 * math.copysign(1, 100 * date.year + date.month - 190002.5) + 0.5
    return julian_datetime
def earthRotationAngle(jd):
    t=jd-2451545
    f=jd%1.0
    theta = 2*math.pi*(f+0.7790572732640+0.00273781191135448*t)
    theta = theta%(2*math.pi)
    if(theta<0):
        theta+=2*math.pi
    return theta
def gmtSiderealTime(date):
    jd = get_julian_datetime(date)
    t = (jd-2451545)/36525
    gmst = earthRotationAngle(jd)+(0.014506+ 4612.156534*t + 1.3915817*t*t - 0.00000044*t*t*t - 0.000029956*t*t*t*t - 0.0000000368*t*t*t*t*t)/60/60*math.pi/180
    gmst%=2*math.pi
    if(gmst<0):
        gmst+=2*math.pi
    return gmst
def azEltoDec(az, el, lat):
    dec = math.asin(math.sin(lat)*math.sin(el)+math.cos(lat)*math.cos(el)*math.cos(az))
    return dec
def azEltoRa(az, el, lat, lon, date):
    dec = azEltoDec(az, el, lat)
    sinH = -math.sin(az) * math.cos(el) / math.cos(dec)
    cosH = (math.sin(el) - math.sin(lat) * math.sin(dec))/(math.cos(lat) * math.cos(dec))

    LHA = math.atan2(sinH, cosH)
    LST = (gmtSiderealTime(date) + lon) % (2 * math.pi)
    RA = (LST - LHA) % (2 * math.pi)

    return RA
df = pd.read_csv("AzEl.csv")
output = pd.DataFrame(
    {
        "RA":[],
        "Dec":[]
    }
)
#df["Az (Rot)"]
for x in range(len(df["Az (Rot)"])):
    azimuth = math.radians(df["Az (Rot)"][x])
    elevation = math.radians(df["El (Rot)"][x])
    dt=datetime.strptime(cdate+" "+str(df["Time"][x]),"%Y-%m-%d %H:%M:%S")+timedelta(hours=4)
    new_row = pd.DataFrame([{'RA': math.degrees(azEltoRa(azimuth,elevation,latitude,longitude,dt))/15, 'Dec': math.degrees(azEltoDec(azimuth,elevation,latitude))}])
    output = pd.concat([output, new_row], ignore_index=True)
output.to_csv("RA_Dec_output.csv", index=False)
