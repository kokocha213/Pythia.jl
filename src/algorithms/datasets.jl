using CSV, DataFrames

function load_dataset(name::String)
    path = joinpath(@__DIR__, "..", "assets", "$name.csv")
    return CSV.read(path, DataFrame)
end