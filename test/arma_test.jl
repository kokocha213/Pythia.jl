using Random
using NPZ
using Test
using .Pythia

function simulate_arma(rng::AbstractRNG, ϕ::Float64, θ::Float64, σ::Float64, n::Int)
    ε = randn(rng, n) .* σ
    y = zeros(n)
    y[1] = 5
    for t in 2:n
        y[t] = ϕ * y[t-1] + ε[t] + θ * ε[t-1]
    end
    return y 
end

preds_py = npzread("test/python_testing/preds.npy")
conf_int_py = npzread("test/python_testing/conf_int.npy")  

rng = MersenneTwister(40)
ϕ = 0.6
θ = 0.3
σ = 1.0
y = simulate_arma(rng, ϕ, θ, σ, 2000)

y_train = y[1:1995]
y_test  = y[1996:2000]

model = ARIMAModel(y_train; p=1, q=1, P=0, Q=0, seasonal=false, stationary=true,approximation = false)
fitted_model = fit(model)

@test begin
    _, pval = check_residuals(fitted_model, plot_diagnostics=false)
    pval > 0.5
end

res   = predict(fitted_model, h=5, level=0.95)
preds = res.fittedvalues
lower = res.lower
upper = res.upper

@test all(abs.(lower .- conf_int_py[:,1]) .≤ 0.01)
@test all(abs.(upper .- conf_int_py[:,2]) .≤ 0.01)
@test all(abs.(preds .- preds_py) .≤ 0.01)
@test all(abs.(y_test .- clamp.(y_test, lower, upper)) .≤ 1e-6)