using Test
using Pythia
import Pythia: kpss_test
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

@testset "internal differencing function" begin 
    y = [10, 12, 14, 13, 15, 17, 16,   # Week 1
        13, 15, 17, 16, 18, 20, 19,   # Week 2
        16, 18, 20, 19, 21, 23, 22]   # Week 3

    l = Pythia.difference_series_(y; D=1, s=7)
    # Test that the differenced series is (almost) zero
    @test all(isapprox.(l.y_diff, 0.0; atol=1e-8))
end

@testset "kpss testing" begin
    sun_data = load_dataset("sunspots")
    years = round.(Int, sun_data.Year)
    ssn   = sun_data.SSN
    result = kpss_test(ssn)
    y = difference(ssn)
    result_diff = kpss_test(y.series)
    @test result_diff.pvalue > 0.5
end