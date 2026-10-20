# Plots RMSE data from calculate_sep.py

import numpy as np
import  matplotlib.pyplot as plt

d = np.load("data/mc_sep_results.npz")
SEP_BINS, crb, SNR = d["sep_bins"], d["crb"], d["snr"]
rmse = {name: d[f"{name}_t1"] for name in ["FIIB", "Root-MUSIC", "ESPRIT"]}

# ~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots()
ax.plot(SEP_BINS, crb, "k--", label="CRB")
for name, values in rmse.items():
    ax.plot(SEP_BINS, values, label=f"{name}")
ax.set_yscale("log")
ax.set_xlabel("Range Separation (bins)")
ax.set_ylabel("Range RMSE (bins)")
ax.set_title(f"Target 1, SNR = {SNR} dB")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()