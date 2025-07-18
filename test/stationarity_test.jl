using Test
import .Pythia: kpss_test  # or Pythia.Stationarity: kpss_test if inside submodule
using Random
Random.seed!(42)

# --- Data Generators ---
function generate_stationary_series(n=200, ϕ=0.6)
    y = zeros(n)
    for t in 2:n
        y[t] = ϕ * y[t-1] + randn()
    end
    return y
end

function generate_trend_stationary_series(n=200, α=0.5)
    return α .* (1:n) .+ randn(n)
end

function generate_unit_root_series(n=200)
    return cumsum(randn(n))
end

# -----------------------
@testset "KPSS Tests: Stationary Series" begin
    y = generate_stationary_series()

    for reg in ("c", "ct"), lag in ("auto", "legacy")
        result = kpss_test(y, regression=reg, nlags=lag)
        @test result.statistic < result.critical_values["5%"]  # should not reject
        @test result.pvalue > 0.05
    end
end

@testset "KPSS Tests: Unit Root (Random Walk)" begin
    y = generate_unit_root_series()

    for reg in ("c", "ct"), lag in ("auto", "legacy")
        result = kpss_test(y, regression=reg, nlags=lag)
        @test result.statistic > result.critical_values["10%"]  # should reject
        @test result.pvalue < 0.1
    end
end
