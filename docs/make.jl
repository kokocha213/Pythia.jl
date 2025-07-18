using Documenter
using Pythia

makedocs(
    sitename = "Pythia.jl",
    modules = [YourPackageName],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        # add more pages if needed
    ]
)
