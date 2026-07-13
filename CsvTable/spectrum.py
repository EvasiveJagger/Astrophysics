"""
Loader for the trailing spectral data in <date>-Spectrum.csv.

Structure discovered by inspecting the file:
  - 33 named metadata columns, then a "Data" column header.
  - Each row carries 1536 trailing numeric values (plus one empty field from a
    trailing comma), laid out as three stacked 512-bin blocks:

      block 0  bins    0-511   raw FFT power, linear   (sum == "Power (FFT)" column)
      block 1  bins  512-1023  normalized spectrum     (== block 2 / 365.86)
      block 2  bins 1024-1535  Tsys per channel, KELVIN (mean == "Tsys" column)

    Block 2 is the calibrated one -- use it unless you specifically want raw counts.

  - Frequency axis: "Centre Freq" (Hz) with "Bandwidth" (Hz) spread over FFT_SIZE
    bins. For this dataset that is 2.2275 GHz +/- 128 kHz, i.e. ~500 Hz per bin.
"""

import numpy as np
import pandas as pd

FFT_SIZE = 512
N_BLOCKS = 3

# which block to treat as "the" spectrum
BLOCK_RAW = 0
BLOCK_NORMALIZED = 1
BLOCK_KELVIN = 2


def load_spectra(path, block=BLOCK_KELVIN):
    """
    Read the CSV and return (meta, spectra, freq_hz).

      meta     -- DataFrame of the 33 named columns (RA, Dec, Tsys, Power (dBFS), ...)
      spectra  -- (N, 512) float array of the requested block
      freq_hz  -- (512,) float array, the frequency of each bin in Hz

    The file is ragged (metadata columns + 1536 unnamed spectral columns), so it
    is read with header=None and split by position rather than by column name.
    """
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")

    n_meta = header.index("Data")  # 33 -- everything before the spectral payload

    raw = pd.read_csv(
        path,
        header=None,
        skiprows=1,
        usecols=range(n_meta + N_BLOCKS * FFT_SIZE),
        low_memory=False,
    )

    meta = raw.iloc[:, :n_meta]
    meta.columns = header[:n_meta]

    payload = raw.iloc[:, n_meta:].to_numpy(dtype=float)  # (N, 1536)
    blocks = payload.reshape(len(payload), N_BLOCKS, FFT_SIZE)
    spectra = blocks[:, block, :]  # (N, 512)

    centre = float(meta["Centre Freq"].iloc[0])
    bandwidth = float(meta["Bandwidth"].iloc[0])
    bin_width = bandwidth / FFT_SIZE
    freq_hz = centre - bandwidth / 2.0 + (np.arange(FFT_SIZE) + 0.5) * bin_width

    return meta, spectra, freq_hz


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    meta, spectra, freq_hz = load_spectra("krabspec.csv", block=BLOCK_KELVIN)

    freq_mhz = freq_hz / 1e6

    print(f"{len(spectra)} spectra x {spectra.shape[1]} channels")
    print(f"band: {freq_mhz.min():.6f} - {freq_mhz.max():.6f} MHz "
          f"({(freq_hz[1] - freq_hz[0]):.1f} Hz per channel)")

    # sanity check against the named columns -- these should match to rounding
    print(f"mean(block2) vs Tsys column: "
          f"{spectra[0].mean():.4f} vs {float(meta['Tsys'].iloc[0]):.4f}")

    mean_spectrum = spectra.mean(axis=0)
    std_spectrum = spectra.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # averaging all N pointings together beats down uncorrelated noise, so any
    # narrow line or persistent RFI in the band should stand out here
    ax1.plot(freq_mhz, mean_spectrum, lw=0.8)
    ax1.fill_between(
        freq_mhz,
        mean_spectrum - std_spectrum,
        mean_spectrum + std_spectrum,
        alpha=0.25,
    )
    ax1.set_ylabel("Tsys (K)")
    ax1.set_title("mean spectrum over all pointings (band +/- 1 sigma)")

    # a waterfall makes drift and intermittent RFI visible as horizontal streaks
    im = ax2.imshow(
        spectra,
        aspect="auto",
        origin="lower",
        extent=[freq_mhz.min(), freq_mhz.max(), 0, len(spectra)],
        cmap="viridis",
    )
    ax2.set_xlabel("frequency (MHz)")
    ax2.set_ylabel("sample index")
    ax2.set_title("waterfall")
    fig.colorbar(im, ax=ax2, label="Tsys (K)")

    plt.tight_layout()
    plt.savefig("spectrum_overview.png", dpi=150)
    plt.show()
