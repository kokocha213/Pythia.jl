# Stationarity and detrending (ADF/KPSS)

Stationarity means that the statistical properties of a time series i.e. mean, variance and covariance do not change over time. Many statistical models require the series to be stationary to make effective and precise predictions.

Two statistical tests would be used to check the stationarity of a time series – Augmented Dickey Fuller (“ADF”) test and Kwiatkowski-Phillips-Schmidt-Shin (“KPSS”) test. A method to convert a non-stationary time series into stationary series shall also be used.

This first cell imports standard packages and sets plots to appear inline.

```@example station
using Pythia,Plots
```
Sunspots dataset is used. It contains yearly (1700-2008) data on sunspots from the National Geophysical Data Center.

```@example station
sun_data = load_dataset("sunspots")
years = round.(Int, sun_data.Year)
ssn   = sun_data.SSN
```
The data is plotted now.

```@example station
plot(
    years,
    ssn,
    xlabel = "Year",
    ylabel = "Sunspot Number (SSN)",
    title = "Yearly Sunspot Activity",
    lw = 2,
    legend = false,
    size = (1200, 800)
)
```

## Stationarity tests

KPSS is a test for checking the stationarity of a time series.

Null Hypothesis: The process is trend stationary.

Alternate Hypothesis: The series has a unit root (series is not stationary).

```@example station
kpss_test(ssn)
```

Based upon the significance level of 0.05 and the p-value of KPSS test, there is evidence for rejecting the null hypothesis in favor of the alternative. Hence, the series is non-stationary as per the KPSS test.

ADF test is used to determine the presence of unit root in the series, and hence helps in understand if the series is stationary or not. The null and alternate hypothesis of this test are:

Null Hypothesis: The series has a unit root.

Alternate Hypothesis: The series has no unit root.

If the null hypothesis in failed to be rejected, this test may provide evidence that the series is non-stationary.

```@example station
adf_test(ssn)
```
Based upon the significance level of 0.05 and the p-value of ADF test, the null hypothesis can not be rejected. Hence, the series is non-stationary.


It is always better to apply both the tests, so that it can be ensured that the series is truly stationary. Possible outcomes of applying these stationary tests are as follows:

Case 1: Both tests conclude that the series is not stationary - The series is not stationary
Case 2: Both tests conclude that the series is stationary - The series is stationary
Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
Here, due to the difference in the results from ADF test and KPSS test, it can be inferred that the series is trend stationary and not strict stationary. The series can be detrended by differencing or by model fitting.

## Detrending by Differencing
It is one of the simplest methods for detrending a time series. A new series is constructed where the value at the current time step is calculated as the difference between the original observation and the observation at the previous time step.

Differencing is applied on the data and the result is plotted.

```@example station
y = difference(ssn)
ssn_diff = y.series
```

This applies auto differencing on the data till it passes the KPSS test

```@example station
years = years[1:end-1]
plot(
    years,
    ssn_diff,
    xlabel = "Year",
    ylabel = "Sunspot Number (SSN)",
    title = "Yearly Sunspot Activity",
    lw = 2,
    legend = false,
    size = (1200, 800)
)
```
ADF test is now applied on these detrended values and stationarity is checked.

```@example station
adf_test(ssn_diff)
```

Based upon the p-value of ADF test, the series is strict stationary now.

KPSS test is now applied on these detrended values and stationarity is checked.

```@example station
kpss_test(ssn_diff)
```
Based upon the p-value of KPSS test, the null hypothesis can not be rejected. Hence, the series is stationary.

## Conclusion

Two tests for checking the stationarity of a time series are used, namely ADF test and KPSS test. Detrending is carried out by using differencing. Trend stationary time series is converted into strict stationary time series. Requisite forecasting model can now be applied on a stationary time series data.