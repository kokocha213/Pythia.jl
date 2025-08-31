using Random
using NPZ
using Distributions
using Pythia

Random.seed!(1234)


@testset "p_value testing" begin 
    n = 100
    phi = 0.5
    σ = 1.0
    ε = rand(Normal(0, σ), n)
    y = zeros(n)
    y[1] = ε[1]

    for t in 2:n
        y[t] = phi * y[t-1] + ε[t]
    end

    model = ARIMAModel(y; p=1, q=0, P=0, Q=0, seasonal=false, stationary=true,approximation = false)
    fitted_model = Pythia.fit(model)

    _, pval = check_residuals(fitted_model, plot_diagnostics=false)

    pval_py = npzread("test/python_testing/p_value.npy")
    pval_py = pval_py[1]

    @test isapprox(pval, pval_py, atol=0.001)
end