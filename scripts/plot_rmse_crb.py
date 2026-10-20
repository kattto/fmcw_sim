# Plots the RMSE data from calculate_rmse_crb.py

import numpy as np
import  matplotlib.pyplot as plt

d = np.load("data/mc_crb_results_og.npz")
SNR_DB, crb = d["snr_db"], d["crb"]
rmse = {name: d[name] for name in ["FIIB", "Root-MUSIC", "ESPRIT"]}

# ~~~~~~~~~~~~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots()
ax.plot(SNR_DB, crb, "k--", label="CRB")
for name, values in rmse.items():
    ax.plot(SNR_DB, values, label=f"{name} RMSE")
ax.set_yscale("log")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Range RMSE (bins)")
# ax.set_title(f"asd")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.show()