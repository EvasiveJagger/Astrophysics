from astropy.table import Table

t = Table.read("spag.fits")

print("CTYPE2:", t["CTYPE2"][0])
print("CTYPE3:", t["CTYPE3"][0])
print("CRVAL2 sample:", t["CRVAL2"][:5])
print("CRVAL3 sample:", t["CRVAL3"][:5])
print("TRGTLONG sample:", t["TRGTLONG"][:5])
print("TRGTLAT sample:", t["TRGTLAT"][:5])
print("DATA shape:", t["DATA"].shape)
print("TDIM7:", t["TDIM7"][0])
print("TUNIT7:", t["TUNIT7"][0])
print("DATA sample row:", t["DATA"][0][:20])   # first 20 channel values of first row