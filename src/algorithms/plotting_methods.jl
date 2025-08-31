using Plots
using StatsBase
using StatsPlots
using HypothesisTests
using Distributions

"""
    plot_vectors(vectors::Vector{<:AbstractVector}, 
                 labels::Vector{String}, 
                 colors::Vector{Symbol}; 
                 linestyles::Union{Nothing, Vector{Symbol}} = nothing)

Plot multiple vectors on the same plot. 

Arguments:
- `vectors`: A list of vectors to plot.
- `labels`: Labels for each line (used in the legend).
- `colors`: Line colors for each vector.

Optional keyword:
- `linestyles`: Optional line styles (e.g., `:solid`, `:dash`, `:dot`, etc.)
"""
function plot_vectors(vectors::Vector{<:AbstractVector}, 
                      labels::Vector{String}, 
                      colors::Vector{Symbol}; 
                      linestyles::Union{Nothing, Vector{Symbol}} = nothing)

    @assert length(vectors) == length(labels) == length(colors) "All inputs must have the same length"
    if linestyles !== nothing
        @assert length(linestyles) == length(vectors) "Line styles must match the number of vectors"
    end

    plt = plot()
    for i in eachindex(vectors)
        ls = linestyles === nothing ? :solid : linestyles[i]
        plot!(plt, vectors[i], label=labels[i], color=colors[i], linestyle=ls)
    end

    display(plt)
end
"""
    check_residuals(model; lag=24, plot_diagnostics=true)

Diagnostic checks for ARIMA residuals.

# Arguments
- `model`: a fitted ARIMA model with residuals stored in `model.fitted_model.residuals`.

# Keyword Arguments
- `lag::Int = 24`: maximum lag for the ACF and Ljung-Box test.
- `plot_diagnostics::Bool = true`: whether to generate diagnostic plots.

# Returns
- `plt::Plots.Plot` (or `nothing` if `plot_diagnostics=false`): the combined residual diagnostic plot.
- `pval::Float64`: Ljung–Box p-value for the residuals.

This function does the following:

1. Plots residual diagnostics (if `plot_diagnostics=true`):
   - Residuals vs time
   - ACF of residuals with 95% confidence bands
   - Histogram of residuals with fitted Normal distribution overlay
2. Computes the Ljung–Box test for white noise in residuals.
"""
function check_residuals(model; lag=24, plot_diagnostics=true)
    if model.fitted_model === nothing
        error("Model has not been fitted. Call fit() first.")
    end

    residuals = model.fitted_model.residuals
    n = length(residuals)

    # Ljung–Box test
    lbq = LjungBoxTest(residuals, lag)
    df = lbq.lag - lbq.dof
    pval = 1 - cdf(Chisq(df), lbq.Q)

    if !plot_diagnostics
        return nothing, pval
    else
        # Residuals vs time
        p1 = plot(residuals, seriestype=:line, title="Residuals", legend=false, lw=1.5)

        # ACF of residuals
        acf_vals = autocor(residuals, 1:lag)
        conf = 1.96 / sqrt(n)  # 95% confidence bands
        p2 = bar(1:lag, acf_vals, title="ACF of Residuals", legend=false)
        hline!(p2, [conf, -conf], color=:blue, linestyle=:dash)

        # Histogram with Normal overlay
        μ̂, σ̂ = mean(residuals), std(residuals)
        dist = Normal(μ̂, σ̂)
        p3 = histogram(residuals, normalize=:pdf, bins=:auto, title="Residuals Distribution", legend=false)
        plot!(p3, x -> pdf(dist, x), color=:red, lw=2)

        # Print Ljung–Box results
        println(lbq)

        # Combine plots
        plt = plot(p1, p2, p3, layout=(3,1), size=(800,600))

        # Return plot and p-value
        return plt, pval
    end
end
"""
    plot_acf_pacf(y; lags=20, alpha=0.05) -> Plot

Compute and plot the autocorrelation function (ACF) and partial autocorrelation function (PACF) 
for a univariate time series, including approximate confidence bounds.

# Arguments
- `y::AbstractVector{<:Real}`: Time series data

# Keyword Arguments
- `lags::Int=20`: Number of lags to compute and display for ACF and PACF
- `alpha::Float64=0.05`: Significance level for confidence bounds (default 95% confidence)

# Returns
- `Plot`: Combined plot with ACF on top and PACF below, including confidence bounds

# Examples
```julia
using Random
Random.seed!(123)
y = randn(100)

# Default 20 lags with 95% confidence bounds
plot_acf_pacf(y)

# Custom number of lags
plot_acf_pacf(y, lags=30, alpha=0.01)
```

# Notes
- ACF includes lag 0, while PACF typically starts from lag 1.
- The plots can help detect:
  - Autoregressive (AR) structure: PACF cuts off after lag p
  - Moving average (MA) structure: ACF cuts off after lag q

# References
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). *Time Series Analysis: Forecasting and Control*.
- Brockwell, P.J., & Davis, R.A. (2016). *Introduction to Time Series and Forecasting*.
"""
function plot_acf_pacf(y::AbstractVector{<:Real}; lags::Int=20, alpha::Float64=0.05)
    n = length(y)
    z = quantile(Normal(), 1 - alpha/2)
    conf = z ./ sqrt(n)

    acf_vals = autocor(y, 0:lags)
    pacf_vals = pacf(y, 1:lags)

    p1 = bar(0:lags, acf_vals, legend=false, xlabel="Lag", ylabel="ACF",
             title="Autocorrelation Function (ACF)")
    hline!(p1, [conf, -conf], linestyle=:dash, color=:red)

    p2 = bar(1:lags, pacf_vals, legend=false, xlabel="Lag", ylabel="PACF",
             title="Partial Autocorrelation Function (PACF)")
    hline!(p2, [conf, -conf], linestyle=:dash, color=:red)
    
    plot(p1, p2, layout=(2,1), size=(800,500))
end