# Autoregressive Moving Average (ARMA): Artificial data

In this example, we will:

1. Generate artificial ARMA(1,1) data.  
2. Split the data into training and test sets.  
3. Fit an ARIMA(1,0,1) model (equivalent to ARMA(1,1)).  
4. Forecast the next 5 values with 95% prediction intervals.

---

## Step 1. Simulate ARMA(1,1) data

We define a helper function to generate an ARMA(1,1) series:

```@example arma
using Random, Pythia

function simulate_arma(rng::AbstractRNG, ϕ::Float64, θ::Float64, σ::Float64, n::Int)
    ε = randn(rng, n) .* σ
    y = zeros(n)
    y[1] = 5
    for t in 2:n
        y[t] = ϕ * y[t-1] + ε[t] + θ * ε[t-1]
    end
    return y
end
```

Now generate 2000 data points with parameters ϕ = 0.6, θ = 0.3, and σ = 1:

```@example arma
using Random
rng = Random.MersenneTwister(40)
ϕ, θ, σ = 0.6, 0.3, 1.0

y = simulate_arma(rng, ϕ, θ, σ, 2000)

length(y)  # confirm we generated 2000 values
```

---

## Step 2. Train-test split

We’ll keep the first 1995 points for training, and the last 5 points for testing:

```@example arma
y_train = y[1:1995]
y_test  = y[1996:2000]

(size(y_train), size(y_test))
```

---

## Step 3. Fit ARIMA(1,0,1) model

An ARMA(1,1) is equivalent to ARIMA with order (p=1, d=0, q=1).  
We fit the model on the training data:

```@example arma
using Pythia
model = Pythia.ARIMAModel(y_train; p=1, q=1, P=0, Q=0, seasonal=false, stationary=true, approximation=false)
fitted_model = fit(model)
```

---

## Step 4. Forecast next 5 values

Finally, we forecast the next 5 points, with 95% prediction intervals:

```@example arma
res   = predict(fitted_model, h=5, level=0.95)
preds = res.fittedvalues
lower = res.lower
upper = res.upper
```
# Forecast Plot (Last 50 points)

```@example arma
using Plots

window = 50
y_last = y_train[end-window+1:end]
x_last = (length(y_train)-window+1):length(y_train)

h = length(preds)
x_forecast = length(y_train)+1 : length(y_train)+h

# Plot last 50 points of y_train
plt = plot(x_last, y_last, label="y_train (last 50)", color=:blue)

# Plot forecast
plot!(plt, x_forecast, preds, label="Forecast", color=:red, lw=2)

# Plot upper and lower as light red lines
plot!(plt, x_forecast, lower, label="Lower 95%", color=:red, alpha=0.3, lw=2, linestyle=:dash)
plot!(plt, x_forecast, upper, label="Upper 95%", color=:red, alpha=0.3, lw=2, linestyle=:dash)

xlabel!("Time")
ylabel!("Value")
title!("Forecast vs Observed (last 50 points)")
plt
```

