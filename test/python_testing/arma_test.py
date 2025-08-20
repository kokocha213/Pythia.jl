# save_preds.py
import numpy as np
import pandas as pd
import pmdarima as pm

# --- load y series saved from Julia ---
df = pd.read_csv("arma_series.csv")   # from Julia
y = df["y"].values

y_train, y_test = y[:1995], y[1995:]

# --- fit ARIMA(1,0,1) with L-BFGS-B ---
model = pm.ARIMA(order=(1,0,1), seasonal_order=(0,0,0,0), method="lbfgs",scoring='mse')
model.fit(y_train)

# --- forecast 5 steps ---
preds, conf_int = model.predict(n_periods=5, return_conf_int=True)

print("y_test:", y_test)
print("preds:", preds)
print("conf_int:", conf_int)

# --- save as .npy ---
np.save("preds.npy", preds)
np.save("conf_int.npy", conf_int)
np.save("y_test.npy", y_test)

print("Saved predictions to preds.npy, conf_int.npy, y_test.npy")
