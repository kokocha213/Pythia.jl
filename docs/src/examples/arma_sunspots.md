
# ARMA Modeling and Residual Diagnostics in Julia

It explores AR models on the sunspots dataset, generates a simulated ARMA(4,1) process, 
and examines residual diagnostics.

## example block: loads all necessary packages and fixes the seed

```@example sun
using Pythia
using Plots
using Random
using LinearAlgebra

Random.seed!(1234) 
```

## Load and visualize sunspots data

```@example sun
sun_data = load_dataset("sunspots")
years = round.(Int, sun_data.Year)
ssn   = sun_data.SSN
```

```@example sun
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

## ACF and PACF

```@example sun
plot_acf_pacf(ssn)
```

## Fit AR(2) model

```@example sun
model_ar2 = Pythia.ARIMAModel(ssn; p=2, q=0, seasonal=false, stationary=true, approximation=false)
fitted_ar2 = fit(model_ar2)
get_ic(fitted_ar2)
plt, pval = check_residuals(fitted_ar2)
plt
```

## Fit AR(3) model

```@example sun
model_ar3 = Pythia.ARIMAModel(ssn; p=3, q=0, seasonal=false, stationary=true, approximation=false)
fitted_ar3 = fit(model_ar3)
get_ic(fitted_ar3)
plt, pval = check_residuals(fitted_ar3)
plt
```

## Function to generate ARMA process

```@example sun
function generate_arma(arparams::Vector{Float64}, maparams::Vector{Float64}, n::Int; burnin=250)
    p = length(arparams) - 1
    q = length(maparams) - 1

    φ = -arparams[2:end]  # statsmodels convention
    θ =  maparams[2:end]

    N = n + burnin
    y = zeros(Float64, N)
    ε = randn(N)

    for t in 1:max(p, q, N)
        if p > 0 && t > p
            y[t] += dot(φ, y[t-1:-1:t-p])
        end
        if q > 0 && t > q
            y[t] += dot(θ, ε[t-1:-1:t-q])
        end
        y[t] += ε[t]
    end

    return y[burnin+1:end]
end
```

## Simulate ARMA(4,1)

```@example sun
arparams = [1.0, 0.35, -0.15, 0.55, 0.1]
maparams = [1.0, 0.65]
sim_data = generate_arma(arparams, maparams, 500)
```

```@example sun
plot_acf_pacf(sim_data)
```

## Fit ARMA(1,1) (incorrect) on simulated data

```@example sun
model_arma11 = Pythia.ARIMAModel(sim_data; p=1, q=1, seasonal=false, stationary=true, approximation=false)
fitted_arma11 = fit(model_arma11)
plt, pval = check_residuals(fitted_arma11)
plt
```

## Fit ARMA(4,1) (correct) on simulated data

```@example sun
model_arma41 = Pythia.ARIMAModel(sim_data; p=4, q=1, seasonal=false, stationary=true, approximation=false)
fitted_arma41 = fit(model_arma41)
plt, pval = check_residuals(fitted_arma41)
plt
```
