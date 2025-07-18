using Pythia
using Test

function simulate_arma(ϕ::Float64, θ::Float64, σ::Float64, n::Int)
    ε = randn(n) .* σ
    y = zeros(n)
    y[1] = 5
    for t in 2:n
        y[t] = ϕ * y[t-1] + ε[t] + θ * ε[t-1]
    end
    trend = 0.2 .* collect(1:n)
    seasonal = 5.0 .* sin.((2π / 7) .* collect(1:n))
    return y .+ trend .+ seasonal
end

ϕ = 0.6
θ = 0.3
σ = 0.0
y = simulate_arma(ϕ, θ, σ, 200)

y_train = y[1:195]
model = ARIMAModel(y_train; p=1, q=1, d=1, P=0, Q=0, D=1, s=7, stationary=false)
fitted_model = fit(model)

y_pred = predict(fitted_model)
y_predicted = y_pred[196:200]
y_actual = y[196:200]

# === Test that predictions are accurate ===
@test all(isapprox.(y_predicted, y_actual; atol=1e-5))
