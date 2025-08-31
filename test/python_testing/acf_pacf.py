import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

# ---------------------------
# 1. Read CSV data
# ---------------------------
df = pd.read_csv("ar1_sample.csv")
y = df["Value"].values

# ---------------------------
# 2. Compute ACF and PACF
# ---------------------------
nlags = 20  # Number of lags to compute
acf_vals = acf(y, nlags=nlags, fft=False)
pacf_vals = pacf(y, nlags=nlags)

# ---------------------------
# 3. Save to .npy files
# ---------------------------
np.save("acf_vals.npy", acf_vals)
np.save("pacf_vals.npy", pacf_vals)

print("Saved ACF and PACF values to acf_vals.npy and pacf_vals.npy")
