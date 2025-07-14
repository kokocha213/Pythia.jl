using HypothesisTests, Statistics
using LinearAlgebra, Statistics, Distributions
export difference, seasonal_difference
using LinearAlgebra, Statistics, Distributions

"""
    kpss_test(y::AbstractVector{<:Real};
              regression::String = "c",
              nlags::Union{String, Int} = "auto")

Perform the KPSS test for level or trend stationarity.

# Arguments
- `y`: Time series data (1D vector).
- `regression`: `"c"` for constant, `"ct"` for trend.
- `nlags`: `"auto"` (default, Hobijn-style), `"legacy"` (Schwert), or an integer.

# Returns
NamedTuple with:
- `statistic`: KPSS test statistic
- `pvalue`: Approximate p-value
- `lags`: Lag used
- `critical_values`: Dict of critical values at 10%, 5%, 1%
"""
function kpss_test(y::AbstractVector{<:Real};
                   regression::Symbol = :c,
                   nlags::Union{String, Int} = "auto")

    n = length(y)
    @show n

    # 1. Detrend
    if regression == :ct
        X = [ones(n) (1:n)]
        β = X \ y
        residuals = y - X * β
        @show β
        @show mean(residuals), std(residuals)
    elseif regression == :c
        residuals = y .- mean(y)
        @show mean(residuals), std(residuals)
    else
        error("regression must be either \"c\" (constant) or \"ct\" (trend)")
    end

    # 2. Partial sums
    S = cumsum(residuals)
    @show mean(S), std(S)

    # 3. Lag selection
    lags = if nlags == "auto"
        Int(floor(4 * (n / 100)^(2 / 9)))  # Hobijn approx
    elseif nlags == "legacy"
        Int(floor(12 * (n / 100)^(1 / 4)))  # Schwert
    elseif isa(nlags, Int)
        nlags
    else
        error("nlags must be \"auto\", \"legacy\", or an integer")
    end
    @show lags

    # 4. Long-run variance
    ω² = sum(residuals .^ 2) / n
    for k in 1:lags
        γ_k = sum(residuals[1:end-k] .* residuals[k+1:end]) / n
        ω² += 2 * (1 - k / (lags + 1)) * γ_k
    end
    @show ω²

    # 5. Test statistic
    η = sum(S .^ 2) / (n^2 * ω²)
    @show η

    # 6. Critical values
    crit = regression == :ct ?
        Dict("10%" => 0.119, "5%" => 0.146, "1%" => 0.216) :
        Dict("10%" => 0.347, "5%" => 0.463, "1%" => 0.739)

    # 7. P-value (rough)
    pval = regression == :ct ? 
        1 - cdf(Chisq(1), η * 100) :
        1 - cdf(Chisq(2), η * 25)
    @show pval

    return (
        statistic = η,
        pvalue = pval,
        lags = lags,
        critical_values = crit
    )
end

function difference(y::Vector{T}; 
                    d::Union{Nothing, Int}=nothing,
                    alpha::Float64=0.05,
                    max_d::Int=2,
                    test::Symbol=:kpss,
                    trend::Symbol=:c) where T <: AbstractFloat

    test ∈ [:kpss, :adf] || error("Test must be :kpss, :adf, or :pp")
    alpha > 0 && alpha < 1 || error("Alpha must be between 0 and 1")
    max_d ≥ 0 || error("max_d must be ≥ 0")

    y_diff = copy(y)

    # === Manual differencing if `d` given ===
    if d !== nothing
        for _ in 1:d
            y_diff = diff(y_diff)
        end
        return (series = y_diff, d_applied = d, stationary = true)
    end

    # === Auto-differencing via tests ===
    d_auto = 0
    stationary = false
    test_stats = Float64[]

    for i in 0:max_d
        if i > 0
            y_diff = diff(y_diff)
        end
        length(y_diff) < 10 && break

        if test == :kpss
            test_result = kpss_test(y_diff; regression = trend)
            push!(test_stats, test_result.statistic)
            stationary = test_result.pvalue > alpha

        elseif test == :adf
            test_result = HypothesisTests.ADFTest(y_diff, trend ? "ct" : "c")
            push!(test_stats, test_result.stat)
            stationary = pvalue(test_result) <= alpha
        end

        if stationary
            d_auto = i
            break
        elseif i == max_d
            d_auto = max_d
            @warn "Maximum differences ($max_d) reached – series may still be non-stationary"
        end
    end

    return (series = y_diff,d_applied=d_auto, stationary = stationary)
end

function seasonal_difference(y::AbstractVector{<:Real}, m::Int; 
                             D::Union{Nothing, Int}=nothing,
                             test::Symbol=:ocsb,  # :ch or :ocsb
                             alpha::Float64=0.05,
                             max_D::Int=2,
                             trend::Bool=true)

    y_diff = copy(y)

    # === Manual seasonal differencing if D given ===
    for _ in 1:D
        y_diff = y_diff[(m+1):end] .- y_diff[1:(end - m)]
    end
    return (series = y_diff, D_applied = D, test_results = nothing)
end
