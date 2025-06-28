module Pythia

using Base: Integer, Float64, sign_mask
using Statistics, Distributions
using Optim
using LinearAlgebra

export sglasso, glasso, lasso
export MeanForecast, NaiveForecast, SES, Holt, HoltWinters
export ARIMAModel
export fit, predict
export difference, auto_seasonal_difference

include("algorithms/arma.jl")
include("algorithms/sglasso.jl")
include("algorithms/basicMethods.jl")
include("algorithms/ETS.jl")
include("algorithms/Stationarity.jl")

end
