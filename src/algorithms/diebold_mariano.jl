using Statistics
using Distributions

"""
    dm_test(actual, pred1, pred2; h=1, crit="MSE", power=2)

Performs the Diebold-Mariano (DM) test to compare forecast accuracy between two prediction methods.

# Arguments
- `actual::Vector{<:Real}`: The actual observed values.
- `pred1::Vector{<:Real}`: The predicted values from model 1.
- `pred2::Vector{<:Real}`: The predicted values from model 2.

# Keyword Arguments
- `h::Int`: The forecast horizon (must be > 0 and < length of the input vectors).
- `crit::String`: Criterion to evaluate forecast errors. One of `"MSE"`, `"MAD"`, `"MAPE"`, or `"poly"`.
- `power::Real`: Used only when `crit="poly"`, exponent to apply to the error.

# Returns
A NamedTuple: `(DM = test_statistic, p_value = p_value)`
"""
function dm_test(actual::Vector{<:Real}, pred1::Vector{<:Real}, pred2::Vector{<:Real};
                 h::Int=1, crit::String="MSE", power::Real=2)

    # Input validation
    n = length(actual)
    if length(pred1) != n || length(pred2) != n
        error("Length of actual, pred1, and pred2 must be equal.")
    end
    if h < 1 || h ≥ n
        error("Forecast horizon h must be ≥ 1 and less than the length of the data.")
    end
    if !(crit in ["MSE", "MAD", "MAPE", "poly"])
        error("crit must be one of: \"MSE\", \"MAD\", \"MAPE\", \"poly\".")
    end

    # Calculate error differentials d_t
    d = Float64[]
    for i in eachindex(actual)
        a, f1, f2 = actual[i], pred1[i], pred2[i]
        e1, e2 = 0.0, 0.0
        if crit == "MSE"
            e1 = (a - f1)^2
            e2 = (a - f2)^2
        elseif crit == "MAD"
            e1 = abs(a - f1)
            e2 = abs(a - f2)
        elseif crit == "MAPE"
            if a == 0
                error("Cannot compute MAPE with actual value of 0 at index $i.")
            end
            e1 = abs((a - f1) / a)
            e2 = abs((a - f2) / a)
        elseif crit == "poly"
            e1 = (a - f1)^power
            e2 = (a - f2)^power
        end
        push!(d, e1 - e2)
    end

    # Mean of loss differential
    d̄ = mean(d)

    # Autocovariance function
    function autocov(d::Vector{Float64}, lag::Int)
        T = length(d)
        μ = mean(d)
        sum((d[i+lag] - μ) * (d[i] - μ) for i in 1:(T - lag)) / T
    end

    T = length(d)
    γ = [autocov(d, j) for j in 0:h-1]
    V_d = (γ[1] + 2*sum(γ[2:end])) / T
    if V_d ≤ 0
        error("Variance estimate V_d is non-positive (=$V_d). Try a smaller forecast horizon `h` or use a larger dataset.")
    end
    # DM statistic with Harvey adjustment
    DM_stat = d̄ / sqrt(V_d)
    harvey_adj = sqrt((T + 1 - 2h + h*(h - 1)/T) / T)
    DM_stat *= harvey_adj

    # Two-tailed p-value from Student's t-distribution
    p_value = 2 * cdf(TDist(T - 1), -abs(DM_stat))

    return (DM = DM_stat, p_value = p_value)
end
