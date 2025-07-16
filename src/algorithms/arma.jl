using Optim
import Optim: optimize
mutable struct ARIMAModel
    y::AbstractVector{<:Real}
    h::Int

    p::Union{Nothing, Int}
    d::Union{Nothing, Int}
    q::Union{Nothing, Int}
    P::Union{Nothing, Int}
    D::Union{Nothing, Int}
    Q::Union{Nothing, Int}
    s::Union{Nothing, Int}

    max_p::Int
    max_q::Int
    max_P::Int
    max_Q::Int
    max_d::Int
    max_D::Int

    auto::Bool
    stepwise::Bool
    approximation::Bool
    stationary::Bool
    trace::Bool 
    best_ic::Union{Nothing, Float64}

    seasonal::Bool
    ic::String
    loss::Function
    optimizer::Function

    fitted_model::Union{Nothing, Any}
end

"""
    ARIMAModel( y; h=5, p=nothing, d=nothing, q=nothing,
                 P=nothing, D=nothing, Q=nothing, s=nothing,
                 seasonal=true, ic="aic",
                 loss=sse_arima_loss_, optimizer = () -> Optim.LBFGS(),
                 stationary=false, trace=false,
                 stepwise=true, approximation=true,
                 max_p=5, max_q=5, max_P=2, max_Q=2, max_d=2, max_D=1,
                 auto=true)

Keyword‑based convenience constructor that forwards to the
positional (generic) constructor Julia created automatically.
Only the first argument `y` is required.
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
    get_ic(model::ARIMAModel)
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

    # print(model.optimizer)
    # Step 4: Optimize
    result = Optim.optimize(loss_fn, θ0, model.optimizer())
    # print(result)

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

function fit_arima_(model::ARIMAModel)
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
        # Bail out if outside the user’s limits
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


function inverse_difference(forecasts::Vector{Float64}, y_last::Float64, d::Int)
    inv_forecasts = copy(forecasts)
    for d_iter in 1:d
        inv_forecasts = cumsum([y_last; inv_forecasts])[2:end]
    end
    return inv_forecasts
end
function predict(model::ARIMAModel; h::Int = model.h)
    if model.fitted_model === nothing
        error("Model has not been fitted. Call fit_arima_manual_ first.")
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
    include_intercept = (model.d + model.D) < 2
    μ = include_intercept ? θ[1] : 0.0
    offset = include_intercept ? 1 : 0

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
                ma_term += θ[offset + p + j] * residuals[n + t - j]
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
    
    return vcat(model.y, inv_forecasts)
end
