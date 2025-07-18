using Plots

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
