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

    seasonal::Bool
    ic::String
    loss::Function
    optimizer::Function
    stationary::Bool

    fitted_model::Union{Nothing, Any}

    function ARIMAModel(y::AbstractVector{<:Real}; h::Int = 5,
                        p::Union{Nothing, Int} = nothing,
                        d::Union{Nothing, Int} = nothing,
                        q::Union{Nothing, Int} = nothing,
                        P::Union{Nothing, Int} = nothing,
                        D::Union{Nothing, Int} = nothing,
                        Q::Union{Nothing, Int} = nothing,
                        s::Union{Nothing, Int} = nothing,
                        seasonal::Bool = true,
                        ic::String = "aic",
                        loss::Function = sse_arima_loss_,
                        optimizer::Function = () -> Optim.LBFGS(),
                        stationary::Bool = false)

        if isempty(y)
            error("Input time series `y` must not be empty.")
        end
        if h <= 0
            error("Forecast horizon `h` must be a positive integer.")
        end
        if !(ic in ["aic", "bic", "aicc"])
            error("Information criterion must be one of: \"aic\", \"bic\", or \"aicc\".")
        end

        if any(x -> x === nothing, [p, q])
            error("Parameters `p` and `q` must be specified for ARIMA.")
        end

        if seasonal
            if any(x -> x === nothing, [P, Q, s])
                error("Seasonal parameters `P`, `Q`, and `s` must be specified when `seasonal = true`.")
            end
            if s > 1 && D === nothing
                error("Seasonal differencing order `D` must be specified when seasonal period `s > 1`.")
            end
        end

        return new(y, h, p, d, q, P, D, Q, s, seasonal, ic, loss, optimizer, stationary, nothing)
    end
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
    end
    return (y_diff = y_diff,d=d_result.d_applied, D=D_result.D_applied,y_diff_naive=y_diff_naive)
end
function sse_arima_loss_(y::AbstractVector{<:Real}, θ::Vector{Float64},
                        p::Int, q::Int, P::Int, Q::Int, s::Int;
                        return_residuals::Bool = false)

    n = length(y)
    total_p = p + P
    total_q = q + Q

    φ = θ[1:total_p]             # AR + SAR
    θ_ma = θ[total_p+1:end]      # MA + SMA

    e = zeros(n)

    for t in 1:n
        ar_term = 0.0
        ma_term = 0.0

        # AR terms
        for i in 1:p
            lag = t - i
            if lag >= 1
                ar_term += φ[i] * y[lag]
            end
        end

        # Seasonal AR terms
        for i in 1:P
            lag = t - i * s
            if lag >= 1
                ar_term += φ[p + i] * y[lag]
            end
        end

        # MA terms
        for j in 1:q
            lag = t - j
            if lag >= 1
                ma_term += θ_ma[j] * e[lag]
            end
        end

        # Seasonal MA terms
        for j in 1:Q
            lag = t - j * s
            if lag >= 1
                ma_term += θ_ma[q + j] * e[lag]
            end
        end

        # Residual
        e[t] = y[t] - ar_term - ma_term
    end

    return return_residuals ? e : sum(e.^2)
end

function fit(model::ARIMAModel)
    # Step 1: Apply differencing
    if !model.stationary
        y=difference_series_(model.y; d=model.d, D=model.D, s=model.s).y_diff
    else y=model.y
    end
    # Step 2: Setup
    total_p = model.p + (model.P isa Int ? model.P : 0)
    total_q = model.q + (model.Q isa Int ? model.Q : 0)
    θ0 = zeros(total_p + total_q)

    # Step 3: Define loss function
    loss_fn = θ -> model.loss(y, θ, model.p, model.q, model.P, model.Q, model.s)

    # print(model.optimizer)
    # Step 4: Optimize
    result = Optim.optimize(loss_fn, θ0, model.optimizer())
    # print(result)

    # Step 5: Extract results
    params = Optim.minimizer(result)
    loglik = -Optim.minimum(result)
    σ² = sum((model.loss(y, params, model.p, model.q, model.P, model.Q, model.s; return_residuals=true)).^2) / length(y)

    # Step 6: Store result
    model.fitted_model = (
        params = params,
        loglik = loglik,
        sigma2 = σ²,
        residuals = model.loss(y, params, model.p, model.q, model.P, model.Q, model.s; return_residuals=true)
    )
    
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
        error("Model has not been fitted. Call fit_arma! first.")
    end
    θ = model.fitted_model.params
    p, q = model.p, model.q
    P, Q, s = model.P, model.Q, model.s
    is_seasonal = model.seasonal
    D=0
    d=0
    if !model.stationary
        ds=difference_series_(model.y; d=model.d, D=model.D, s=model.s)
        y_diff = ds.y_diff_naive
        y=ds.y_diff
        d=ds.d
        D=ds.D
    else 
        y=model.y
    end
    n = length(y)
    forecasts = zeros(h)
    history = copy(y)
    residuals = model.fitted_model.residuals
    for t in 1:h
        ar_term = 0.0
        ma_term = 0.0

        for i in 1:p
            if n + t - i > 0
                ar_term += θ[i] * history[n + t - i]
            end
        end

        for j in 1:q
            if n + t - j <= length(residuals)
                ma_term += θ[p + j] * residuals[n + t - j]
            end
        end

        forecasts[t] = ar_term + ma_term
        push!(history, forecasts[t]) 
    end

    print(forecasts)
    inv_forecasts = copy(forecasts)
    if !model.stationary
        if D > 0 && s > 0
            y_last_seasonal = y_diff[end - s*D + 1:end]
            print(y_last_seasonal)
            inv_forecasts = inverse_seasonal_difference(inv_forecasts, y_last_seasonal, s, D)
        end
        print(inv_forecasts)
        if d > 0    
            y_last_nonseasonal = model.y[end - d + 1]
            inv_forecasts = inverse_difference(inv_forecasts, y_last_nonseasonal, d)
        end
    end
    full_series = vcat(model.y, inv_forecasts)
    return full_series
end