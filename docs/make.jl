using Documenter
using Pythia

makedocs(
    sitename = "Pythia.jl",
    modules = [Pythia],
    repo = "https://github.com/ababii/Pythia.jl",
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "Examples" => [
            "Autoregressive Moving Average (ARMA): Artificial data" => "examples/arma_generated.md",
            "Autoregressive Moving Average (ARMA): Sunspots data" => "examples/arma_sunspots.md",
            "Stationarity and detrending (ADF/KPSS)" => "examples/stationarity_sun.md",
        ],
        "API Reference" => "api.md",
        # add more pages if needed
    ]
)
