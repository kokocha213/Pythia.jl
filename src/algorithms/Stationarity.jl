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
                   regression::String = "c",
                   nlags::Union{String, Int} = "auto")

    n = length(y)
    # @show n

    # 1. Detrend
    if regression == "ct"
        X = [ones(n) (1:n)]
        β = X \ y
        residuals = y - X * β
        # @show β
        # @show mean(residuals), std(residuals)
    elseif regression == "c"
        residuals = y .- mean(y)
        # @show mean(residuals), std(residuals)
    else
        error("regression must be either \"c\" (constant) or \"ct\" (trend)")
    end

    # 2. Partial sums
    S = cumsum(residuals)
    # @show mean(S), std(S)

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
    # @show lags

    # 4. Long-run variance
    ω² = sum(residuals .^ 2) / n
    for k in 1:lags
        γ_k = sum(residuals[1:end-k] .* residuals[k+1:end]) / n
        ω² += 2 * (1 - k / (lags + 1)) * γ_k
    end
    # @show ω²

    # 5. Test statistic
    η = sum(S .^ 2) / (n^2 * ω²)
    # @show η

    # 6. Critical values
    crit = regression == "ct" ?
        Dict("10%" => 0.119, "5%" => 0.146, "1%" => 0.216) :
        Dict("10%" => 0.347, "5%" => 0.463, "1%" => 0.739)

    # 7. P-value (rough)
    pval = regression == "ct" ? 
        1 - cdf(Chisq(1), η * 100) :
        1 - cdf(Chisq(2), η * 25)
    # @show pval

    return (
        statistic = η,
        pvalue = pval,
        lags = lags,
        critical_values = crit
    )
end


function adf_test(y::AbstractVector{<:Real};
                  regression::String="c",
                  max_lags::Int=10,
                  criterion::Symbol=:AIC)

    # Determine trend term
    n = length(y)
    trend_column = regression == "c" ? ones(n-1) : hcat(ones(n-1), collect(2:n))  
    best_ic = Inf
    best_test = nothing
    best_lag = 0

    Δy_full = diff(y)

    for lags in 0:max_lags
        # dependent variable: Δy_t starting from (lags+1)
        Y = Δy_full[(lags+1):end]

        # independent variables
        X = y[(lags+1):(end-1)]  # y_{t-1}
        X = reshape(X, :, 1)

        # add lagged differences
        for i in 1:lags
            Xlag = Δy_full[(lags+1-i):(end-i)]
            X = hcat(X, Xlag)
        end

        # add trend/constant
        X = hcat(trend_column[(lags+1):end, :], X)

        # OLS
        β = X \ Y
        residuals = Y - X * β
        σ2 = mean(residuals.^2)
        k = size(X, 2)
        llf = -0.5 * length(Y) * (log(2π*σ2) + 1)
        ic = criterion == :AIC ? -2*llf + 2*k : -2*llf + log(length(Y))*k

        if ic < best_ic
            best_ic = ic
            best_lag = lags
            best_test = HypothesisTests.ADFTest(y, regression=="c" ? :constant : :trend , lags)
        end
    end

    return (
        statistic = best_test.stat,
        pvalue = pvalue(best_test),
        lags = best_lag,
        criterion_value = best_ic
    )
end


function difference(y::AbstractVector{<:Real}; 
                    d::Union{Nothing, Int}=nothing,
                    alpha::Float64=0.05,
                    max_d::Int=2,
                    test::String="kpss",
                    trend::String="c") 

    test ∈ ["kpss", "adf"] || error("Test must be \"kpss\", \"adf\", or \"pp\"")
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

        if test == "kpss"
            test_result = kpss_test(y_diff; regression = trend)
            push!(test_stats, test_result.statistic)
            stationary = test_result.pvalue > alpha

        elseif test == "adf"
            test_result = adf_test(y_diff, regression = trend)
            push!(test_stats, test_result.statistic)
            stationary = test_result.pvalue <= alpha
        end

        if stationary
            d_auto = i
            break
        elseif i == max_d
            d_auto = max_d
            @warn "Maximum differences ($max_d) reached – series may still be non-stationary"
        end
    end

    return (series = y_diff, d_applied = d_auto, stationary = stationary)
end

function seasonal_difference(y::AbstractVector{<:Real}, m::Int; 
                             D::Union{Nothing, Int}=nothing,
                             test::String="ocsb",  # "ch" or "ocsb"
                             alpha::Float64=0.05,
                             max_D::Int=2,
                             trend::Bool=true)

    y_diff = copy(y)

    # === Manual seasonal differencing if D given ===
    if D !== nothing
        for _ in 1:D
            y_diff = y_diff[(m+1):end] .- y_diff[1:(end - m)]
        end
    end

    return (series = y_diff, D_applied = D, test_results = nothing)
end

