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
    check_residuals(model; lag=24)

Diagnostic checks for ARIMA residuals.

# Arguments
- `model`: a fitted ARIMA model with residuals stored in `model.fitted_model.residuals`.

# Keyword Arguments
- `lag::Int = 24`: maximum lag for the ACF and Ljung-Box test.
- `plot_diagnostics::Bool = true`: to plot residual diagnostics.

This function plots residual diagnostics:
1. Residuals vs time
2. ACF of residuals
3. Histogram + Normal density overlay

It also runs a Ljung-Box test for white noise in residuals.
"""
function check_residuals(model; lag=24, plot_diagnostics=true)
    if model.fitted_model === nothing
        error("Model has not been fitted. Call fit() first.")
    end
    
    residuals = model.fitted_model.residuals
    n = length(residuals)
    if (!plot_diagnostics)
        lbq = LjungBoxTest(residuals, lag)
        println(lbq)
        df = lbq.lag - lbq.dof
        pval = 1 - cdf(Chisq(df), lbq.Q)
        return pval
    else
        p1 = plot(residuals, seriestype=:line, title="Residuals", legend=false, lw=1.5)

        acf_vals = autocor(residuals, 1:lag)
        conf = 1.96 / sqrt(n)  # 95% confidence bands
        p2 = bar(1:lag, acf_vals, title="ACF of Residuals", legend=false)
        hline!(p2, [conf, -conf], color=:blue, linestyle=:dash)

        μ̂, σ̂ = mean(residuals), std(residuals)
        dist = Normal(μ̂, σ̂)
        p3 = histogram(residuals, normalize=:pdf, bins=:auto, title="Residuals Distribution", legend=false)
        plot!(p3, x -> pdf(dist, x), color=:red, lw=2)

        lbq = LjungBoxTest(residuals, lag)
        println(lbq)
        
        plot(p1, p2, p3, layout=(3,1), size=(800,600))
        df = lbq.lag - lbq.dof
        pval = 1 - cdf(Chisq(df), lbq.Q)
        return pval
    end
end