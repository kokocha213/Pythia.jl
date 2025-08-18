using Optim
import Optim: optimize

"""
    ARIMAModel

A comprehensive implementation of AutoRegressive Integrated Moving Average (ARIMA) models 
with seasonal extensions (SARIMA), based on the Hyndman-Khandakar algorithm.

This implementation supports both automatic and manual model specification, with various 
optimization strategies and information criteria for model selection.

# Mathematical Background

ARIMA(p,d,q) × (P,D,Q)s models are defined by:
- φ(B)(1-B)^d Φ(B^s)(1-B^s)^D y_t = θ(B)Θ(B^s)ε_t + μ

Where:
- φ(B) = 1 - φ₁B - φ₂B² - ... - φₚBᵖ (non-seasonal AR polynomial)
- Φ(B^s) = 1 - Φ₁B^s - Φ₂B^{2s} - ... - ΦₚB^{Ps} (seasonal AR polynomial)  
- θ(B) = 1 + θ₁B + θ₂B² + ... + θₑB^q (non-seasonal MA polynomial)
- Θ(B^s) = 1 + Θ₁B^s + Θ₂B^{2s} + ... + ΘₖB^{Qs} (seasonal MA polynomial)
- B is the backshift operator
- ε_t ~ N(0,σ²) are white noise errors

# References
- Hyndman, R.J., & Khandakar, Y. (2008). "Automatic time series forecasting: the forecast package for R". Journal of Statistical Software, 26(3), 1-22.
- Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015). "Time Series Analysis: Forecasting and Control". 5th Edition.
"""
mutable struct ARIMAModel
    y::AbstractVector{<:Real}           # Time series data
    h::Int                              # Forecast horizon

    # ARIMA orders
    p::Union{Nothing, Int}              # Non-seasonal AR order
    d::Union{Nothing, Int}              # Non-seasonal differencing order
    q::Union{Nothing, Int}              # Non-seasonal MA order
    P::Union{Nothing, Int}              # Seasonal AR order
    D::Union{Nothing, Int}              # Seasonal differencing order
    Q::Union{Nothing, Int}              # Seasonal MA order
    s::Union{Nothing, Int}              # Seasonal period

    # Search constraints
    max_p::Int                          # Maximum non-seasonal AR order
    max_q::Int                          # Maximum non-seasonal MA order
    max_P::Int                          # Maximum seasonal AR order
    max_Q::Int                          # Maximum seasonal MA order
    max_d::Int                          # Maximum non-seasonal differencing
    max_D::Int                          # Maximum seasonal differencing

    # Algorithm settings
    auto::Bool                          # Enable automatic model selection
    stepwise::Bool                      # Use stepwise search (Hyndman-Khandakar)
    approximation::Bool                 # Use conditional sum of squares approximation
    stationary::Bool                    # Assume series is stationary
    trace::Bool                         # Enable optimization tracing
    best_ic::Union{Nothing, Float64}    # Best information criterion value found

    # Model configuration
    seasonal::Bool                      # Enable seasonal components
    ic::String                          # Information criterion ("aic", "aicc", "bic")
    loss::Function                      # Loss function for optimization
    optimizer::Function                 # Optimization algorithm factory

    # Fitted model storage
    fitted_model::Union{Nothing, Any}   # Fitted model parameters and statistics
end

"""
    ARIMAModel(y; kwargs...)

Create an ARIMA model with automatic parameter selection capabilities.

This constructor implements the Hyndman-Khandakar algorithm for automatic ARIMA model 
selection, which combines unit root tests, information criteria, and stepwise search 
to find optimal model parameters.

# Arguments
- `y::AbstractVector{<:Real}`: Time series data (required)

# Keyword Arguments
- `h::Int=5`: Forecast horizon (number of periods to forecast)
- `p::Union{Nothing,Int}=nothing`: Non-seasonal AR order (auto-determined if nothing)
- `d::Union{Nothing,Int}=nothing`: Non-seasonal differencing order (auto-determined if nothing)
- `q::Union{Nothing,Int}=nothing`: Non-seasonal MA order (auto-determined if nothing)
- `P::Union{Nothing,Int}=nothing`: Seasonal AR order (auto-determined if nothing)
- `D::Union{Nothing,Int}=nothing`: Seasonal differencing order (auto-determined if nothing)
- `Q::Union{Nothing,Int}=nothing`: Seasonal MA order (auto-determined if nothing)
- `s::Union{Nothing,Int}=nothing`: Seasonal period (e.g., 12 for monthly data)
- `seasonal::Bool=true`: Enable seasonal modeling
- `ic::String="aicc"`: Information criterion ("aic", "aicc", "bic")
- `loss::Function=sse_arima_loss_`: Loss function for parameter estimation
- `optimizer::Function=() -> Optim.LBFGS()`: Optimization algorithm
- `stationary::Bool=false`: Assume series is already stationary
- `trace::Bool=false`: Print optimization progress
- `stepwise::Bool=true`: Use stepwise search (recommended for speed)
- `approximation::Bool=true`: Use CSS approximation for speed
- `max_p::Int=5`: Maximum non-seasonal AR order to consider
- `max_q::Int=5`: Maximum non-seasonal MA order to consider
- `max_P::Int=2`: Maximum seasonal AR order to consider
- `max_Q::Int=2`: Maximum seasonal MA order to consider
- `max_d::Int=2`: Maximum non-seasonal differencing order
- `max_D::Int=1`: Maximum seasonal differencing order
- `auto::Bool=true`: Enable automatic model selection

# Examples
```julia
# Basic automatic ARIMA for quarterly data
y = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]
model = ARIMAModel(y; s=4)  # s=4 for quarterly seasonality

# Manual ARIMA(1,1,1) specification
model = ARIMAModel(y; p=1, d=1, q=1, auto=false)

# High-frequency data with custom settings
daily_data = rand(365) .+ sin.(2π * (1:365) / 365)  # Daily data with annual cycle
model = ARIMAModel(daily_data; 
                   s=365, 
                   h=30,           # 30-day forecast
                   max_p=3,        # Limit AR terms
                   max_q=3,        # Limit MA terms
                   approximation=false,  # Use exact ML
                   ic="bic")       # Use BIC for selection

# Non-seasonal ARIMA with custom optimizer
model = ARIMAModel(y; 
                   seasonal=false, 
                   optimizer=() -> Optim.NelderMead(),
                   trace=true)
```

# Mathematical Details
The constructor implements the Hyndman-Khandakar algorithm:
1. Determine differencing orders (d, D) using unit root tests
2. Select initial parameter values from a restricted set
3. Use stepwise search to explore neighboring parameter combinations
4. Select final model based on information criterion

# References
- Hyndman, R.J., & Khandakar, Y. (2008). "Automatic time series forecasting: the forecast package for R"
"""
function ARIMAModel(y::AbstractVector{<:Real};
                      h::Int = 5,
                      p::Union{Nothing,Int}=nothing,
                      d::Union{Nothing,Int}=nothing,
                      q::Union{Nothing,Int}=nothing,
                      P::Union{Nothing,Int}=nothing,
                      D::Union{Nothing,Int}=nothing,
                      Q::Union{Nothing,Int}=nothing,
                      s::Union{Nothing,Int}=nothing,
                      seasonal::Bool = true,
                      ic::String = "aicc",
                      loss::Function = sse_arima_loss_,
                      optimizer::Function = () -> Optim.LBFGS(),
                      stationary::Bool = false,
                      trace::Bool = false,
                      stepwise::Bool = true,
                      approximation::Bool = true,
                      max_p::Int = 5,  max_q::Int = 5,
                      max_P::Int = 2,  max_Q::Int = 2,
                      max_d::Int = 2,  max_D::Int = 1,
                      auto::Bool = true)

    return ARIMAModel(
        y, h, p, d, q, P, D, Q, s,
        max_p, max_q, max_P, max_Q, max_d, max_D,
        auto, stepwise, approximation, stationary, trace, nothing,
        seasonal, ic, loss, optimizer,
        nothing
    )
end

"""
    get_ic(model::ARIMAModel) -> Float64

Calculate the corrected Akaike Information Criterion (AICc) for a fitted ARIMA model.

The AICc is particularly useful for small samples and is defined as:
AICc = AIC + (2k(k+1))/(n-k-1)

Where:
- AIC = -2 * log-likelihood + 2k
- k = number of parameters
- n = sample size

# Arguments
- `model::ARIMAModel`: A fitted ARIMA model

# Returns
- `Float64`: The AICc value (lower is better)

# Examples
```julia
# Fit model and get AICc
model = ARIMAModel(y; p=1, d=1, q=1)
fitted_model = fit(model)
aicc_value = get_ic(fitted_model)
println("AICc: ", aicc_value)
```

# Mathematical Formula
AICc = -2ℓ + 2k + (2k(k+1))/(n-k-1)

Where ℓ is the maximized log-likelihood.

# References
- Hurvich, C.M., & Tsai, C.L. (1989). "Regression and time series model selection in small samples"
"""
function get_ic(model::ARIMAModel)
    # Extract required fields
    cand_model = model.fitted_model
    loglik = cand_model.loglik
    σ2     = cand_model.sigma2
    n      = length(cand_model.residuals)
    k      = length(cand_model.params)  # Number of estimated parameters

    return -2 * loglik + 2k + (2k*(k+1)) / (n - k - 1)
end

"""
    difference_series_(y; kwargs...) -> NamedTuple

Apply non-seasonal and seasonal differencing to achieve stationarity.

This function implements the differencing component of the Hyndman-Khandakar algorithm,
using unit root tests to determine appropriate differencing orders.

# Arguments
- `y::AbstractVector{<:Real}`: Time series data

# Keyword Arguments
- `d::Union{Nothing,Int}=nothing`: Non-seasonal differencing order (auto-determined if nothing)
- `D::Union{Nothing,Int}=nothing`: Seasonal differencing order (auto-determined if nothing)
- `s::Union{Nothing,Int}=nothing`: Seasonal period
- `alpha::Float64=0.05`: Significance level for unit root tests
- `max_d::Int=2`: Maximum non-seasonal differencing order
- `max_D::Int=2`: Maximum seasonal differencing order
- `test::Symbol=:kpss`: Unit root test for non-seasonal differencing
- `seasonal_test::Symbol=:ocsb`: Unit root test for seasonal differencing
- `trend::Bool=true`: Include trend in unit root tests

# Returns
- `NamedTuple`: Contains `y_diff` (final differenced series), `d` (applied non-seasonal order), 
  `D` (applied seasonal order), `y_diff_naive` (after non-seasonal differencing only)

# Examples
```julia
# Automatic differencing for monthly data
y = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]
result = difference_series_(y; s=12)
println("Applied d=", result.d, ", D=", result.D)

# Manual differencing specification
result = difference_series_(y; d=1, D=1, s=12)
differenced_series = result.y_diff

# Non-seasonal differencing only
result = difference_series_(y; d=1, s=nothing)
```

# Mathematical Background
- Non-seasonal differencing: ∇ᵈyₜ = (1-B)ᵈyₜ
- Seasonal differencing: ∇ˢᴰyₜ = (1-Bˢ)ᴰyₜ
- Combined: ∇ᵈ∇ˢᴰyₜ = (1-B)ᵈ(1-Bˢ)ᴰyₜ

# References
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root"
"""
function difference_series_(y::AbstractVector{<:Real};
                            d::Union{Nothing, Int} = nothing,
                            D::Union{Nothing, Int} = nothing,
                            s::Union{Nothing, Int} = nothing,
                            alpha::Float64 = 0.05,
                            max_d::Int = 2,
                            max_D::Int = 2,
                            test::Symbol = :kpss,
                            seasonal_test::Symbol = :ocsb,
                            trend::Bool = true)

    # 1. Apply non-seasonal differencing
    d_result = difference(y;
                          d = d,
                          alpha = alpha,
                          max_d = max_d,
                          test = test)
    
    y_diff_naive = d_result.series

    # 2. Apply seasonal differencing if seasonal period is specified
    D_result = nothing
    if s !== nothing
        D_result = seasonal_difference(y_diff_naive, s;
                                       D = D,
                                       alpha = alpha,
                                       max_D = max_D,
                                       test = seasonal_test,
                                       trend = trend)
        y_diff = D_result.series
        return (y_diff = y_diff,d=d_result.d_applied, D=D_result.D_applied,y_diff_naive=y_diff_naive)
    end
    return (y_diff = y_diff_naive,
                d = d_result.d_applied,
                D = 0,                   
                y_diff_naive = y_diff_naive)
end

"""
    css_arima_loss_(y, φ, θq, Φ, Θ, μ, s; return_residuals=false) -> Float64 or Vector{Float64}

Calculate the Conditional Sum of Squares (CSS) loss for ARIMA model parameters.

This is a fast approximation to maximum likelihood estimation that conditions on 
initial values and computes the sum of squared residuals. It's particularly useful 
for initial parameter estimation and model comparison.

# Arguments
- `y::AbstractVector{<:Real}`: Time series data
- `φ::Vector{Float64}`: Non-seasonal AR coefficients [φ₁, φ₂, ..., φₚ]
- `θq::Vector{Float64}`: Non-seasonal MA coefficients [θ₁, θ₂, ..., θₑ]
- `Φ::Vector{Float64}`: Seasonal AR coefficients [Φ₁, Φ₂, ..., Φₚ]
- `Θ::Vector{Float64}`: Seasonal MA coefficients [Θ₁, Θ₂, ..., Θₖ]
- `μ::Float64`: Intercept/mean parameter
- `s::Int`: Seasonal period

# Keyword Arguments
- `return_residuals::Bool=false`: Return residuals instead of sum of squares

# Returns
- `Float64`: Sum of squared residuals (if return_residuals=false)
- `Vector{Float64}`: Residual series (if return_residuals=true)

# Examples
```julia
# Example for ARIMA(1,1,1)×(1,1,1)₁₂
y = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]
φ = [0.5]        # AR(1) coefficient
θq = [0.3]       # MA(1) coefficient  
Φ = [0.2]        # Seasonal AR(1) coefficient
Θ = [0.1]        # Seasonal MA(1) coefficient
μ = 0.0          # Mean (usually 0 for differenced data)
s = 12           # Monthly seasonality

# Calculate loss
loss = css_arima_loss_(y, φ, θq, Φ, Θ, μ, s)
println("CSS Loss: ", loss)

# Get residuals
residuals = css_arima_loss_(y, φ, θq, Φ, Θ, μ, s; return_residuals=true)
```

# Mathematical Formula
The CSS approximation computes residuals as:
eₜ = yₜ - μ - Σᵢ₌₁ᵖ φᵢyₜ₋ᵢ - Σᵢ₌₁ᵖ Φᵢyₜ₋ᵢₛ - Σⱼ₌₁ᵈ θⱼeₜ₋ⱼ - Σⱼ₌₁ᵈ Θⱼeₜ₋ⱼₛ

Loss = Σₜ₌ₘ₊₁ⁿ eₜ²

Where m = max(p, q, P×s, Q×s) is the maximum lag.

# References
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). "Time Series Analysis: Forecasting and Control"
"""
function css_arima_loss_(y::AbstractVector{<:Real},
                         φ::Vector{Float64},
                         θq::Vector{Float64},
                         Φ::Vector{Float64},
                         Θ::Vector{Float64},
                         μ::Float64,
                         s::Int; return_residuals::Bool=false)

    yc = @. y - μ                     # centre once
    n, p, q, P, Q = length(yc), length(φ), length(θq), length(Φ), length(Θ)
    maxlag = maximum((p, q, P*s, Q*s, 1))
    e = zeros(n)

    @inbounds for t in maxlag+1:n
        ar_term = 0.0
        ma_term = 0.0
        @simd for i in 1:p            ; ar_term += φ[i] * yc[t-i]          ; end
        @simd for i in 1:P            ; ar_term += Φ[i] * yc[t-i*s]        ; end
        @simd for j in 1:q            ; ma_term += θq[j] * e[t-j]          ; end
        @simd for j in 1:Q            ; ma_term += Θ[j] * e[t-j*s]         ; end
        e[t] = yc[t] - ar_term - ma_term
    end

    return return_residuals ? @view(e[maxlag+1:end]) :
                             sum(abs2, @view e[maxlag+1:end])
end

"""
    sse_arima_loss_(y, φ, θq, Φ, Θ, μ, s; return_residuals=false) -> Float64 or Vector{Float64}

Calculate the Sum of Squared Errors (SSE) loss for ARIMA model parameters.

This function provides exact maximum likelihood estimation by computing the full 
likelihood function. It's more accurate than CSS but computationally more intensive.

# Arguments
- `y::AbstractVector{<:Real}`: Time series data
- `φ::Vector{Float64}`: Non-seasonal AR coefficients [φ₁, φ₂, ..., φₚ]
- `θq::Vector{Float64}`: Non-seasonal MA coefficients [θ₁, θ₂, ..., θₑ]
- `Φ::Vector{Float64}`: Seasonal AR coefficients [Φ₁, Φ₂, ..., Φₚ]
- `Θ::Vector{Float64}`: Seasonal MA coefficients [Θ₁, Θ₂, ..., Θₖ]
- `μ::Float64`: Intercept/mean parameter
- `s::Int`: Seasonal period

# Keyword Arguments
- `return_residuals::Bool=false`: Return residuals instead of sum of squares

# Returns
- `Float64`: Sum of squared residuals (if return_residuals=false)
- `Vector{Float64}`: Residual series (if return_residuals=true)

# Examples
```julia
# Example for ARIMA(2,1,1) model
y = cumsum(randn(100)) + 0.1*(1:100)  # Random walk with drift
φ = [0.3, -0.2]   # AR(2) coefficients
θq = [0.4]        # MA(1) coefficient
Φ = Float64[]     # No seasonal AR
Θ = Float64[]     # No seasonal MA
μ = 0.1           # Small drift
s = 1             # No seasonality

# Calculate exact loss
loss = sse_arima_loss_(y, φ, θq, Φ, Θ, μ, s)
println("SSE Loss: ", loss)

# Get residuals for diagnostic checking
residuals = sse_arima_loss_(y, φ, θq, Φ, Θ, μ, s; return_residuals=true)
```

# Mathematical Formula
The exact likelihood residuals are computed as:
eₜ = yₜ - μ - Σᵢ₌₁ᵖ φᵢyₜ₋ᵢ - Σᵢ₌₁ᵖ Φᵢyₜ₋ᵢₛ - Σⱼ₌₁ᵈ θⱼeₜ₋ⱼ - Σⱼ₌₁ᵈ Θⱼeₜ₋ⱼₛ

For t = 1, 2, ..., n (handling initial conditions appropriately).

# References
- Hamilton, J.D. (1994). "Time Series Analysis"
- Brockwell, P.J., & Davis, R.A. (2016). "Introduction to Time Series and Forecasting"
"""
function sse_arima_loss_(y::AbstractVector{<:Real},
                         φ::Vector{Float64},           # AR coefficients
                         θq::Vector{Float64},          # MA coefficients
                         Φ::Vector{Float64},           # Seasonal AR
                         Θ::Vector{Float64},           # Seasonal MA
                         μ::Float64,                   # Intercept / drift
                         s::Int;                       # Seasonality
                         return_residuals::Bool = false)

    n = length(y)
    p = length(φ)
    q = length(θq)
    P = length(Φ)
    Q = length(Θ)

    e = zeros(n)

    for t in 1:n
        ar_term = 0.0
        ma_term = 0.0

        # Non-seasonal AR terms
        for i in 1:p
            lag = t - i
            if lag ≥ 1
                ar_term += φ[i] * y[lag]
            end
        end

        # Seasonal AR terms
        for i in 1:P
            lag = t - i * s
            if lag ≥ 1
                ar_term += Φ[i] * y[lag]
            end
        end

        # Non-seasonal MA terms
        for j in 1:q
            lag = t - j
            if lag ≥ 1
                ma_term += θq[j] * e[lag]
            end
        end

        # Seasonal MA terms
        for j in 1:Q
            lag = t - j * s
            if lag ≥ 1
                ma_term += Θ[j] * e[lag]
            end
        end

        # Residual
        e[t] = y[t] - μ - ar_term - ma_term
    end

    return return_residuals ? e : sum(e .^ 2)
end

"""
    fit_arima_manual_(model::ARIMAModel) -> ARIMAModel

Fit ARIMA model parameters using nonlinear optimization.

This function estimates ARIMA parameters by minimizing the specified loss function 
using the configured optimization algorithm. It handles parameter constraints and 
computes model statistics.

# Arguments
- `model::ARIMAModel`: ARIMA model with specified orders and configuration

# Returns
- `ARIMAModel`: The same model with fitted parameters stored in `fitted_model`

# Examples
```julia
# Fit a specific ARIMA(1,1,1) model
y = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]
model = ARIMAModel(y; p=1, d=1, q=1, auto=false)
fitted_model = fit_arima_manual_(model)

# Access fitted parameters
params = fitted_model.fitted_model.params
println("Fitted parameters: ", params)
println("Log-likelihood: ", fitted_model.fitted_model.loglik)
println("Residual variance: ", fitted_model.fitted_model.sigma2)

# Fit with custom optimizer
model = ARIMAModel(y; p=2, d=1, q=1, 
                   optimizer=() -> Optim.NelderMead(),
                   auto=false)
fitted_model = fit_arima_manual_(model)
```

# Mathematical Details
The function optimizes the parameter vector θ = [μ, φ₁, ..., φₚ, θ₁, ..., θₑ, Φ₁, ..., Φₚ, Θ₁, ..., Θₖ]
to minimize the loss function L(θ).

The intercept μ is included only when (d + D) < 2.

# References
- Nocedal, J., & Wright, S. (2006). "Numerical Optimization"
"""
function fit_arima_manual_(model::ARIMAModel)
    include_intercept = (model.d + model.D) < 2

    total_p = model.p + (model.P isa Int ? model.P : 0)
    total_q = model.q + (model.Q isa Int ? model.Q : 0)

    nc = include_intercept ? 1 : 0
    θ0 = zeros(total_p + total_q + nc)

    # Step 3: Define loss function
    loss_fn = θ -> begin
        idx = 1
        μ = include_intercept ? θ[idx] : 0.0
        idx += nc

        φ = θ[idx : idx + model.p - 1]; idx += model.p
        θq = θ[idx : idx + model.q - 1]; idx += model.q
        Φ = θ[idx : idx + (model.P isa Int ? model.P : 0) - 1]; idx += model.P isa Int ? model.P : 0
        Θ = θ[idx : end]

        model.loss(model.y, φ, θq, Φ, Θ, μ, model.s)  # pass μ now
    end

    # Step 4: Optimize
    result = Optim.optimize(loss_fn, θ0, model.optimizer())

    # Step 5: Extract results
    params = Optim.minimizer(result)
    
    idx = 1
    μ = include_intercept ? params[idx] : 0.0
    idx += include_intercept ? 1 : 0

    φ  = model.p isa Int ? params[idx : idx + model.p - 1]                     : Float64[]
    idx += model.p isa Int ? model.p : 0

    θq = model.q isa Int ? params[idx : idx + model.q - 1]                     : Float64[]
    idx += model.q isa Int ? model.q : 0

    Φ  = model.P isa Int ? params[idx : idx + model.P - 1]                     : Float64[]
    idx += model.P isa Int ? model.P : 0

    Θ  = model.Q isa Int ? params[idx : idx + model.Q - 1]                     : Float64[]
    idx += model.Q isa Int ? model.Q : 0

    # Compute residuals
    residuals = model.loss(model.y, φ, θq, Φ, Θ, μ, model.s; return_residuals=true)
    n   = length(residuals)
    σ² = sum(residuals .^ 2) / n   
    loglik = -0.5 * n * (log(2π) + 1 + log(σ²))

    # Step 6: Store result
    model.fitted_model = (
        params = params,
        loglik = loglik,
        sigma2 = σ²,
        residuals = residuals
    )
    return model
end

"""
    fit(model::ARIMAModel) -> ARIMAModel

Fit ARIMA model using the Hyndman-Khandakar automatic selection algorithm.

This is the main fitting function that implements the complete automatic ARIMA 
algorithm, including differencing, parameter search, and model selection based 
on information criteria.

# Arguments
- `model::ARIMAModel`: ARIMA model configuration

# Returns
- `ARIMAModel`: Fitted model with optimal parameters

# Examples
```julia
# Automatic ARIMA for monthly data
y = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]
model = ARIMAModel(y; s=12)
fitted_model = fit(model)

println("Selected model: ARIMA(", fitted_model.p, ",", fitted_model.d, ",", fitted_model.q, ")")
println("Seasonal: (", fitted_model.P, ",", fitted_model.D, ",", fitted_model.Q, ")[", fitted_model.s, "]")
println("AICc: ", fitted_model.best_ic)

# Manual specification with fitting
model = ARIMAModel(y; p=1, d=1, q=1, auto=false)
fitted_model = fit(model)

# Custom settings for high-frequency data
daily_data = randn(365) .+ 0.1 * sin.(2π * (1:365) / 365)
model = ARIMAModel(daily_data; 
                   s=7,              # Weekly seasonality
                   stepwise=false,   # Full grid search
                   approximation=false,  # Exact ML
                   trace=true)       # Show progress
fitted_model = fit(model)
```

# Algorithm Steps (Hyndman-Khandakar)
1. Determine differencing orders (d, D) using unit root tests
2. Select starting models from a restricted set
3. Use stepwise search around the best starting model
4. Select final model based on information criterion (AICc)

# References
- Hyndman, R.J., & Khandakar, Y. (2008). "Automatic time series forecasting: the forecast package for R"
"""
function fit(model::ARIMAModel)
    if model.seasonal && model.s === nothing
        error("Seasonal parameters `s` must be specified when `seasonal = true`.")
    end
        if isempty(model.y)
        error("Input time series y must not be empty.")
    end
    if model.h <= 0
        error("Forecast horizon h must be a positive integer.")
    end
    
    d=0
    D=0
    if !model.stationary
        differenced_ = difference_series_(model.y; d=model.d, D=model.D, s=model.s)
        y=differenced_.y_diff
        d=differenced_.d
        D=differenced_.D
    else 
        y=model.y
    end

    const_ok = (d + D) < 2

    loss = model.approximation ? css_arima_loss_ : model.loss

    function try_fit(p,q,P,Q; drift=true)
        # Bail out if outside the user's limits
        if any(<(0), (p,q,P,Q)) || p>model.max_p || q>model.max_q ||
           P>model.max_P || Q>model.max_Q || p+q+P+Q==0
            return (Inf, nothing)
        end

        cand = ARIMAModel(model.y; h=model.h,
                          p=p, d=d, q=q, P=P, D=D, Q=Q, s=m,
                          seasonal=(m>1), ic=model.ic,
                          loss = model.approximation ?
                                 css_arima_loss_ : model.loss,
                          optimizer = model.optimizer,
                          stationary = model.stationary)

        # drift toggle (you can store a flag inside cand if you
        # already support it, or absorb it in the mean term)

        fit_arima_manual_(cand)          # your existing exact/CSS fit
        ic_val = get_ic(cand)
        return (ic_val, cand)
    end

    m = model.s === nothing ? 1 : model.s

    fixed_p = model.p !== nothing
    fixed_q = model.q !== nothing
    fixed_P = model.P !== nothing
    fixed_Q = model.Q !== nothing

    if model.seasonal == false || m == 1
        model.P = 0;  fixed_P = true
        model.Q = 0;  fixed_Q = true
    end

    clip(value, fixed::Bool, fixed_val) = fixed ? fixed_val : value

    

    raw_specs = m == 1 ?
        [(2,2,0,0), (0,0,0,0), (1,0,0,0), (0,1,0,0)] :
        [(2,2,1,1), (0,0,0,0), (1,0,1,0), (0,1,0,1)]

    start_specs = Set{NTuple{4,Int}}()  # use a Set to dedup
    for (p0,q0,P0,Q0) in raw_specs
        push!(start_specs,
            (clip(p0,fixed_p,model.p),
            clip(q0,fixed_q,model.q),
            clip(P0,fixed_P,model.P),
            clip(Q0,fixed_Q,model.Q)))
    end

    best_ic   = Inf
    best_cand = nothing
    drifts = const_ok ? (true, false) : (false,)
    
    for (p0,q0,P0,Q0) in start_specs
        for drift in drifts
            ic_val, cand = try_fit(p0,q0,P0,Q0; drift=drift)
            if ic_val < best_ic
                best_ic, best_cand = ic_val, cand
            end
        end
    end

    current_p, current_q = best_cand.p, best_cand.q
    current_P, current_Q = best_cand.P, best_cand.Q
    
    if model.stepwise
        improved = true

        function allowed_neighbours(fixed_p,fixed_q,fixed_P,fixed_Q)
            moves = Tuple{Int,Int,Int,Int}[]

            # Non‑seasonal single steps
            if !fixed_p
                push!(moves, (-1,0,0,0))
                push!(moves, ( 1,0,0,0))
            end
            if !fixed_q
                push!(moves, (0,-1,0,0))
                push!(moves, (0, 1,0,0))
            end

            # Seasonal single steps
            if !fixed_P
                push!(moves, (0,0,-1,0))
                push!(moves, (0,0, 1,0))
            end
            if !fixed_Q
                push!(moves, (0,0,0,-1))
                push!(moves, (0,0,0, 1))
            end

            # Two‑at‑a‑time steps (only if both dimensions are free)
            if !fixed_p && !fixed_q
                push!(moves, (-1,-1,0,0))
                push!(moves, ( 1, 1,0,0))
            end
            if !fixed_P && !fixed_Q
                push!(moves, (0,0,-1,-1))
                push!(moves, (0,0, 1, 1))
            end

            return moves
        end

        # Build just once
        neighbours = allowed_neighbours(fixed_p,fixed_q,fixed_P,fixed_Q)

        while improved
            improved = false

            best_local_ic   = best_ic
            best_local_cand = best_cand

            for (Δp,Δq,ΔP,ΔQ) in neighbours
                p = current_p + Δp
                q = current_q + Δq
                P = current_P + ΔP
                Q = current_Q + ΔQ

                ic_val, cand = try_fit(p,q,P,Q; drift=const_ok)
                if ic_val < best_local_ic
                    best_local_ic, best_local_cand = ic_val, cand
                end
            end

            if best_local_ic < best_ic - 1e-6   # move to the best neighbour
                best_ic, best_cand = best_local_ic, best_local_cand
                current_p, current_q = best_cand.p, best_cand.q
                current_P, current_Q = best_cand.P, best_cand.Q
                improved = true
            end
        end

    else 
        p_range = fixed_p ? [model.p] : 0:model.max_p
        q_range = fixed_q ? [model.q] : 0:model.max_q
        P_range = fixed_P ? [model.P] : 0:model.max_P
        Q_range = fixed_Q ? [model.Q] : 0:model.max_Q

        for p in p_range, q in q_range,P in P_range, Q in Q_range
            for drift in drifts
                ic_val, cand = try_fit(p,q,P,Q; drift=drift)
                if model.trace
                    println("Evaluated p=$p, q=$q → AICc=$ic_val")
                end
                if ic_val < best_ic
                    best_ic, best_cand = ic_val, cand
                end
            end
        end
    end

    model.fitted_model = best_cand.fitted_model

    model.p, model.q = best_cand.p, best_cand.q
    model.P, model.Q = best_cand.P, best_cand.Q
    model.d, model.D = d, D                            # the differencing actually used
    model.s          = m
    model.best_ic    = best_ic
    
    return model
end

"""
    inverse_seasonal_difference(forecasts, y_last, s, D) -> Vector{Float64}

Reverse seasonal differencing transformation to convert forecasts back to original scale.

This function undoes the seasonal differencing operation (1-Bˢ)ᴰ by iteratively 
adding back the seasonal lags.

# Arguments
- `forecasts::Vector{Float64}`: Forecasts in the seasonally differenced scale
- `y_last::Vector{Float64}`: Last s×D observations from the original series
- `s::Int`: Seasonal period
- `D::Int`: Seasonal differencing order

# Returns
- `Vector{Float64}`: Forecasts transformed back to original scale

# Examples
```julia
# Monthly data with one seasonal difference
forecasts = [0.1, 0.2, -0.1, 0.3]  # Forecasts in differenced scale
y_last = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]  # Last 12 months
s = 12  # Monthly seasonality
D = 1   # One seasonal difference

# Transform back to original scale
original_forecasts = inverse_seasonal_difference(forecasts, y_last, s, D)
println("Original scale forecasts: ", original_forecasts)

# Quarterly data with two seasonal differences
forecasts = [0.05, 0.1, -0.02, 0.08]
y_last = [100, 102, 98, 105, 108, 110, 106, 112]  # Last 8 quarters
s = 4
D = 2

original_forecasts = inverse_seasonal_difference(forecasts, y_last, s, D)
```

# Mathematical Formula
For D = 1: yₜ = ∇ₛyₜ + yₜ₋ₛ
For D = 2: yₜ = ∇ₛ²yₜ + 2yₜ₋ₛ - yₜ₋₂ₛ

Where ∇ₛ = (1-Bˢ) is the seasonal difference operator.

# References
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). "Time Series Analysis: Forecasting and Control"
"""
function inverse_seasonal_difference(forecasts::Vector{Float64}, y_last::Vector{Float64}, s::Int, D::Int)
    inv_forecasts = copy(forecasts)
    series = copy(y_last)  # growing base

    for d_iter in 1:D
        temp = Float64[]
        for t in 1:length(inv_forecasts)
            val = inv_forecasts[t] + series[end - s + t]  # get s steps back
            push!(temp, val)
        end
        append!(series, temp)
        inv_forecasts .= temp
    end
    return inv_forecasts
end

"""
    inverse_difference(forecasts, y_last, d) -> Vector{Float64}

Reverse non-seasonal differencing transformation to convert forecasts back to original scale.

This function undoes the non-seasonal differencing operation (1-B)ᵈ by iteratively 
computing cumulative sums.

# Arguments
- `forecasts::Vector{Float64}`: Forecasts in the differenced scale
- `y_last::Float64`: Last observation from the original series
- `d::Int`: Non-seasonal differencing order

# Returns
- `Vector{Float64}`: Forecasts transformed back to original scale

# Examples
```julia
# First difference (d=1)
forecasts = [0.5, 0.3, -0.2, 0.8]  # Forecasts in first-differenced scale
y_last = 120.0  # Last observation from original series
d = 1

# Transform back to original scale
original_forecasts = inverse_difference(forecasts, y_last, d)
println("Original scale forecasts: ", original_forecasts)
# Output: [120.5, 120.8, 120.6, 121.4]

# Second difference (d=2)
forecasts = [0.1, 0.05, -0.02, 0.1]
y_last = 100.0
d = 2

original_forecasts = inverse_difference(forecasts, y_last, d)
```

# Mathematical Formula
For d = 1: yₜ = ∇yₜ + yₜ₋₁ = yₜ₋₁ + Σᵢ₌₁ᵗ ∇yᵢ
For d = 2: yₜ = ∇²yₜ + 2yₜ₋₁ - yₜ₋₂

Where ∇ = (1-B) is the difference operator.

# References
- Hamilton, J.D. (1994). "Time Series Analysis"
"""
function inverse_difference(forecasts::Vector{Float64}, y_last::Float64, d::Int)
    inv_forecasts = copy(forecasts)
    for d_iter in 1:d
        inv_forecasts = cumsum([y_last; inv_forecasts])[2:end]
    end
    return inv_forecasts
end

"""
    predict(model::ARIMAModel; h=model.h,level) -> NamedTuple{(:fittedvalues, :lower, :upper), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}

Generate forecasts from a fitted ARIMA model.

This function produces point forecasts by iteratively applying the ARIMA equation 
and then transforming back to the original scale by reversing any differencing operations.

# Arguments
- `model::ARIMAModel`: A fitted ARIMA model

# Keyword Arguments
- `h::Int=model.h`: Forecast horizon (number of periods to forecast)
- `level::Float64 = 0.95`: Confidence level for the forecast intervals.

# Returns
- `NamedTuple` with:
    - `fittedvalues::Vector{Float64}`: Forecasted values on the original scale.
    - `lower::Vector{Float64}`: Lower bounds of the prediction intervals.
    - `upper::Vector{Float64}`: Upper bounds of the prediction intervals.

# Examples
```julia
# Basic forecasting
y = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120]
model = ARIMAModel(y; s=12, h=6)
fitted_model = fit(model)
results = predict(fitted_model)
forecasts = results.fittedvalues
lower = results.lower
upper = results.upper

# Display original data and forecasts
n = length(y)
println("Historical data: ", y)
println("Forecasts: ", forecasts)

# Custom forecast horizon
long_forecasts = predict(fitted_model; h=12)  # 12-period forecast

# Seasonal forecasting example
monthly_data = [100, 102, 98, 105, 108, 110, 106, 112, 115, 118, 114, 120,
                125, 128, 122, 132, 135, 138, 130, 142, 145, 148, 140, 155]
model = ARIMAModel(monthly_data; s=12, h=6)
fitted_model = fit(model)
results = predict(fitted_model)

# Extract just the forecast values
forecast_values = results.fittedvalues
println("6-month ahead forecasts: ", forecast_values)
```

# Mathematical Details
For ARIMA(p,d,q)×(P,D,Q)ₓ, the h-step ahead forecast is:
ŷₙ₊ₕ = μ + Σᵢ₌₁ᵖ φᵢŷₙ₊ₕ₋ᵢ + Σᵢ₌₁ᵖ Φᵢŷₙ₊ₓ₋ᵢₛ + Σⱼ₌₁ᵈ θⱼeₙ₊ₕ₋ⱼ + Σⱼ₌₁ᵈ Θⱼeₙ₊ₕ₋ⱼₛ

Where:
- Future errors eₙ₊ₖ = 0 for k > 0
- Past values and errors are used when available
- Forecasts are used for future values in AR terms

# References
- Hyndman, R.J., & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice"
- Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). "Time Series Analysis: Forecasting and Control"
"""
function predict(model::ARIMAModel; h::Int = model.h, level::Float64 = 0.95)
    function compute_psi(φ::Vector, θ::Vector, h::Int)
        p, q = length(φ), length(θ)
        ψ = zeros(h+1)
        ψ[1] = 1.0
        for k in 2:(h+1)
            for i in 1:min(p, k-1)
                ψ[k] += φ[i] * ψ[k-i]
            end
            if k-1 <= q
                ψ[k] += θ[k-1]
            end
        end
        return ψ
    end

    if model.fitted_model === nothing
        error("Model has not been fitted. Call fit() first.")
    end
    θ = model.fitted_model.params
    p, q = model.p, model.q
    P, Q, s = model.P, model.Q, model.s
    is_seasonal = model.seasonal
    D, d = 0, 0

    if !model.stationary
        ds = difference_series_(model.y; d=model.d, D=model.D, s=model.s)
        y_diff = ds.y_diff_naive
        y = ds.y_diff
        d, D = ds.d, ds.D
    else
        y = model.y
    end

    n = length(y)
    forecasts = zeros(h)
    history = copy(y)
    residuals = model.fitted_model.residuals
    σ² = model.fitted_model.sigma2
    include_intercept = (model.d + model.D) < 2
    μ = include_intercept ? θ[1] : 0.0
    offset = include_intercept ? 1 : 0
    φ = θ[offset+1 : offset+p]   # AR part
    ϑ = θ[offset+p+1 : offset+p+q]  # MA part
    ψ = compute_psi(φ, ϑ, h)

    for t in 1:h
        ar_term = 0.0
        ma_term = 0.0

        for i in 1:p
            if n + t - i > 0
                ar_term += θ[offset + i] * history[n + t - i]
            end
        end

        for j in 1:q
            if n + t - j <= length(residuals)
                ma_term += ϑ[j] * residuals[n+t-j]
            end
        end

        forecasts[t] = μ + ar_term + ma_term
        push!(history, forecasts[t])
    end

    inv_forecasts = copy(forecasts)
    if !model.stationary
        if D > 0 && s > 0
            y_last_seasonal = y_diff[end - s * D + 1:end]
            inv_forecasts = inverse_seasonal_difference(inv_forecasts, y_last_seasonal, s, D)
        end
        if d > 0
            y_last_nonseasonal = model.y[end - d + 1]
            inv_forecasts = inverse_difference(inv_forecasts, y_last_nonseasonal, d)
        end
    end
    println(inv_forecasts)

    α = 1 - level
    z = quantile(Normal(), 1 - α/2)   # standard normal critical value
    variances = [σ² * sum(ψ[1:k].^2) for k in 1:h]
    std_errs = sqrt.(variances)

    lowers = inv_forecasts .- z .* std_errs
    uppers = inv_forecasts .+ z .* std_errs

    return (
        fittedvalues = inv_forecasts,
        lower = lowers,
        upper = uppers
    )
end