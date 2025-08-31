import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

df = pd.read_csv("ar1_sample.csv")
y = df["Value"].values

model = ARIMA(y, order=(1,0,0))  # AR=1, I=0, MA=0
fitted = model.fit()

residuals = fitted.resid

lags = 24  # Can choose number of lags
lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)

print(f"ARMA(1,0) model Ljung–Box test (lag {lags}):")
print(lb_test)

p_value = lb_test['lb_pvalue'].iloc[0]
print(f"Ljung–Box p-value at lag {lags}: {p_value:.4f}")

np.save("p_value.npy", np.array([p_value]))
print("Saved p-value to p_value.npy")
