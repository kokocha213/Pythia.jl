using Pythia
using Test
function simulate_arma(ϕ::Float64, θ::Float64, σ::Float64, n::Int)
    ε = randn(n) .* σ
    y = zeros(n)
    y[1] = 5
    for t in 2:n
        y[t] = ϕ * y[t-1] + ε[t] + θ * ε[t-1]
    end
    return y 
end

ϕ = 0.6
θ = 0.3
σ = 1.0
y = simulate_arma(ϕ, θ, σ, 2000)
y_train = y[1:1995]
y_test  = y[1996:2000]   # last 5 points

model = ARIMAModel(y_train; p=1, q=1, P=0, Q=0 ,seasonal = false, stationary=true)
fitted_model = fit(model)
@test check_residuals(fitted_model,plot_diagnostics=false) > 0.5

res = predict(fitted_model, h=5, level=0.95)
lower = res.lower
upper = res.upper
@test all(abs.(y_test .- clamp.(y_test, lower, upper)) .≤ 0.01)
