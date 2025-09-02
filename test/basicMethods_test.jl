using Pythia
using Test
 
### Mean Forecast Tests ###
y = [1.0, 2.0, 3.0, 4.0, 5.0]
mdl = MeanForecast(y, h = 5, level = [80, 95])
fittedMdl = fit(mdl)
results = predict(fittedMdl)

@test results.fittedvalues == [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0]
### Will add tests for prediction intervals in later commits.
@test results.lower[1, :] ≈ [1.18980, 1.01703, 1.01703, 1.01703, 1.01703] atol=1e-4 
@test results.upper[1, :] ≈ [4.81019, 4.98296, 4.98296, 4.98296, 4.98296] atol=1e-4

@test results.lower[2, :] ≈ [0.22814, -0.03641, -0.03641, -0.03641, -0.03641] atol=1e-4
@test results.upper[2, :] ≈ [5.7718, 6.0364, 6.0364, 6.0364, 6.0364] atol=1e-4

### Naive Forecast Tests ###
y = [1.0, 2.0, 3.0, 4.0, 5.0]
mdl = NaiveForecast(y, h = 5, level = [80, 95])
fittedMdl = fit(mdl)
results = predict(fittedMdl)

@test results.fittedvalues == [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

@test results.upper ≈ [
    11.407758 11.407758 11.407758 11.407758 11.407758;
    14.799820 14.799820 14.799820 14.799820 14.799820
] atol=1e-5

@test results.lower ≈ [
    -1.407758 -1.407758 -1.407758 -1.407758 -1.407758;
    -4.799820 -4.799820 -4.799820 -4.799820 -4.799820
] atol=1e-5

### Seasonal Naive Model ###

y = [1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]
mdl = SeasonalNaiveForecast(y, h = 5, level = [80, 95])
fittedMdl = fit(mdl)
results = predict(fittedMdl)

@test results.fittedvalues == [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    3.0, 4.0, 5.0, 6.0, 7.0
]

@test results.lower ≈ [
    -0.844655 -1.126206 -1.407758 -1.689309 -1.970861;
    -2.879892 -3.839856 -4.799820 -5.759784 -6.719748
] atol=1e-5

@test results.upper ≈ [
     6.844655  9.126206 11.407758 13.689309 15.970861;
     8.879892 11.839856 14.799820 17.759784 20.719748
] atol=1e-5

