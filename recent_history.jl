# time: 2025-07-14 20:42:24 UTC
# mode: julia
	θ̂ = params[end]                                # single MA coef
# time: 2025-07-14 20:42:24 UTC
# mode: julia
	@printf("\n=== Manual CSS fit (ARMA(1,1)) ===\n")
# time: 2025-07-14 20:42:24 UTC
# mode: julia
	@printf("    μ̂ :  %+8.4f  (should be ~0)\n", μ̂)
# time: 2025-07-14 20:42:24 UTC
# mode: julia
	@printf("    φ̂ :  %+8.4f  (true 0.7)\n",     φ̂)
# time: 2025-07-14 20:42:24 UTC
# mode: julia
	@printf("    θ̂ :  %+8.4f  (true 0.5)\n",     θ̂)
# time: 2025-07-14 20:42:24 UTC
# mode: julia
	@printf("   σ̂² :  %8.4f   (true 1.0)\n",      σ̂²)
# time: 2025-07-14 20:42:26 UTC
# mode: julia
	@printf(" log L :  %8.3f\n\n",                 loglik)
# time: 2025-07-14 20:43:05 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    p = 5, d = 0, q = 2,
	    P = 0, D = 0, Q = 0,
	    s = 1,
	    seasonal = false,
	    stationary = true,
	)
# time: 2025-07-14 20:43:11 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:43:22 UTC
# mode: julia
	params    = model.fitted_model.params
# time: 2025-07-14 20:43:22 UTC
# mode: julia
	σ̂²       = model.fitted_model.sigma2
# time: 2025-07-14 20:43:22 UTC
# mode: julia
	loglik    = model.fitted_model.loglik
# time: 2025-07-14 20:43:32 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:43:32 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:43:33 UTC
# mode: julia
	θ̂ = params[end]                                #
# time: 2025-07-14 20:44:02 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    p = 1, d = 0, q = 1,
	    P = 0, D = 0, Q = 0,
	    s = 1,
	    seasonal = false,
	    stationary = true,
	)
# time: 2025-07-14 20:44:15 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:44:15 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:44:15 UTC
# mode: julia
	θ̂ = params[end]                                #
# time: 2025-07-14 20:44:22 UTC
# mode: julia
	params    = model.fitted_model.params
# time: 2025-07-14 20:44:22 UTC
# mode: julia
	σ̂²       = model.fitted_model.sigma2
# time: 2025-07-14 20:44:22 UTC
# mode: julia
	loglik    = model.fitted_model.loglik
	
	# unpack for nicer display
# time: 2025-07-14 20:44:23 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:44:23 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:44:23 UTC
# mode: julia
	θ̂ = params[end]
# time: 2025-07-14 20:44:38 UTC
# mode: julia
	model.fitted_model
# time: 2025-07-14 20:44:45 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:44:48 UTC
# mode: julia
	params    = model.fitted_model.params
# time: 2025-07-14 20:44:48 UTC
# mode: julia
	σ̂²       = model.fitted_model.sigma2
# time: 2025-07-14 20:44:48 UTC
# mode: julia
	loglik    = model.fitted_model.loglik
	
	# unpack for nicer display
# time: 2025-07-14 20:44:48 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:44:48 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:44:55 UTC
# mode: julia
	θ̂ = params[end]
# time: 2025-07-14 20:45:16 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    p = 1, d = 0, q = 1,
	    P = 0, D = 0, Q = 0,
	    s = 1,
	    seasonal = false,
	    stationary = true,approximation = true
	)
# time: 2025-07-14 20:45:20 UTC
# mode: julia
	params    = model.fitted_model.params
# time: 2025-07-14 20:45:20 UTC
# mode: julia
	σ̂²       = model.fitted_model.sigma2
# time: 2025-07-14 20:45:20 UTC
# mode: julia
	loglik    = model.fitted_model.loglik
	
	# unpack for nicer display
# time: 2025-07-14 20:45:20 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:45:20 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:45:20 UTC
# mode: julia
	θ̂ = params[end]
# time: 2025-07-14 20:45:27 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:45:28 UTC
# mode: julia
	params    = model.fitted_model.params
# time: 2025-07-14 20:45:28 UTC
# mode: julia
	σ̂²       = model.fitted_model.sigma2
# time: 2025-07-14 20:45:28 UTC
# mode: julia
	loglik    = model.fitted_model.loglik
	
	# unpack for nicer display
# time: 2025-07-14 20:45:28 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:45:28 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:45:29 UTC
# mode: julia
	θ̂ = params[end]
# time: 2025-07-14 20:46:25 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    p = 5, d = 0, q = 5,
	    P = 0, D = 0, Q = 0,
	    s = 1,
	    seasonal = false,
	    stationary = true,approximation = true
	)
# time: 2025-07-14 20:46:29 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:46:33 UTC
# mode: julia
	params    = model.fitted_model.params
# time: 2025-07-14 20:46:33 UTC
# mode: julia
	σ̂²       = model.fitted_model.sigma2
# time: 2025-07-14 20:46:33 UTC
# mode: julia
	loglik    = model.fitted_model.loglik
	
	# unpack for nicer display
# time: 2025-07-14 20:46:33 UTC
# mode: julia
	μ̂ = (model.d + model.D) < 2 ? params[1] : 0.0
# time: 2025-07-14 20:46:33 UTC
# mode: julia
	φ̂ = params[(model.d+model.D) < 2 ? 2 : 1]      # first AR coef
# time: 2025-07-14 20:46:33 UTC
# mode: julia
	θ̂ = params[end]
# time: 2025-07-14 20:50:49 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-14 20:51:19 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:51:23 UTC
# mode: julia
	model.p
# time: 2025-07-14 20:51:30 UTC
# mode: julia
	model.q
# time: 2025-07-14 20:52:07 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    seasonal = false,
	    stationary = true,approximation = true,stepwise = true,trace=true
	)
# time: 2025-07-14 20:52:09 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:52:45 UTC
# mode: julia
	y
# time: 2025-07-14 20:52:53 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    seasonal = false,
	    stationary = true,approximation = true,stepwise = true,trace=true
	)
# time: 2025-07-14 20:52:56 UTC
# mode: julia
	model.d
# time: 2025-07-14 20:52:58 UTC
# mode: julia
	model.D
# time: 2025-07-14 20:53:50 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-14 20:53:54 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    seasonal = false,
	    stationary = true,approximation = true,stepwise = true,trace=true
	)
# time: 2025-07-14 20:54:00 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 20:54:05 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 20:54:10 UTC
# mode: julia
	model.d
# time: 2025-07-14 20:54:12 UTC
# mode: julia
	model.D
# time: 2025-07-14 20:54:15 UTC
# mode: julia
	model.p
# time: 2025-07-14 20:54:17 UTC
# mode: julia
	model.q
# time: 2025-07-14 20:55:18 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    seasonal = false,
	    stationary = true,approximation = true,stepwise = false,trace=true
	)
# time: 2025-07-14 20:55:21 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 20:55:30 UTC
# mode: julia
	model.p
# time: 2025-07-14 20:55:32 UTC
# mode: julia
	model.q
# time: 2025-07-14 20:56:09 UTC
# mode: julia
	y  = generate_arma11(5000)
# time: 2025-07-14 20:56:11 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    seasonal = false,
	    stationary = true,approximation = true,stepwise = false,trace=true
	)
# time: 2025-07-14 20:56:13 UTC
# mode: julia
	y  = generate_arma11(5000)
# time: 2025-07-14 20:56:17 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 20:56:23 UTC
# mode: julia
	model.p
# time: 2025-07-14 20:56:24 UTC
# mode: julia
	model.1
# time: 2025-07-14 20:56:26 UTC
# mode: julia
	model.q
# time: 2025-07-14 20:59:27 UTC
# mode: julia
	predict(model)
# time: 2025-07-14 21:02:00 UTC
# mode: julia
	function generate_deterministic_sarima(N;
	        φ::Vector{Float64}  = [0.7],           # p = 1
	        θq::Vector{Float64} = [0.5],           # q = 1
	        Φ::Vector{Float64}  = [0.4],           # P = 1
	        Θ::Vector{Float64}  = [0.3],           # Q = 1
	        s::Int              = 4,               # seasonality
	        y0::Float64         = 1.0,
	        ε0::Float64         = 1.0)
	
	    p,q,P,Q = length.((φ,θq,Φ,Θ))
	    maxlag  = max(p, q, P*s, Q*s)
	    y = zeros(Float64, N + maxlag)
	    ε = zeros(Float64, N + maxlag)
	    y[1] = y0; ε[1] = ε0                # single non‑zero “kick”
	
	    for t in 2:(N + maxlag)
	        # AR side --------------------------------------------------------
	        ar = 0.0
	        for i in 1:p   ; ar +=   φ[i] * y[t-i]        ; end
	        for i in 1:P   ; ar +=   Φ[i] * y[t-i*s]      ; end
	        # MA side --------------------------------------------------------
	        ma = 0.0
	        for j in 1:q   ; ma +=   θq[j] * ε[t-j]       ; end
	        for j in 1:Q   ; ma +=   Θ[j] * ε[t-j*s]      ; end
	        # Deterministic innovations: ε_t = 0   (t ≥ 2)
	        y[t] = ar + ma
	    end
	    return y[(maxlag+1):(maxlag+N)]
	end
# time: 2025-07-14 21:02:11 UTC
# mode: julia
	true_p, true_q, true_P, true_Q, true_s = 1,1,1,1,4     # <-- change at will
# time: 2025-07-14 21:02:11 UTC
# mode: julia
	y_all = generate_deterministic_sarima(2000;
	         φ  = [0.7], θq = [0.5],
	         Φ  = [0.4], Θ  = [0.3],
	         s  = true_s)
# time: 2025-07-14 21:02:11 UTC
# mode: julia
	train  = view(y_all, 1:1995)
# time: 2025-07-14 21:02:11 UTC
# mode: julia
	future = view(y_all, 1996:2000)
# time: 2025-07-14 21:03:52 UTC
# mode: julia
	"""
	    generate_sarima_deterministic(N;
	        φ   = Float64[],         # length p   (non‑seasonal AR)
	        θq  = Float64[],         # length q   (non‑seasonal MA)
	        Φ   = Float64[],         # length P   (seasonal AR)
	        Θ   = Float64[],         # length Q   (seasonal MA)
	        s   = 1,                 # seasonal period
	        y0  = 1.0,               # initial y "kick"
	        ε0  = 1.0)               # initial ε "kick"
	
	Return a vector `y[1:N]` that follows the SARIMA recursion with
	**εₜ = 0 for all t ≥ maxlag+1**.  Works for any (p,q,P,Q,s) ≥ 0.
	"""
	function generate_sarima_deterministic(N;
	        φ   ::Vector{<:Real} = Float64[],   # p
	        θq  ::Vector{<:Real} = Float64[],   # q
	        Φ   ::Vector{<:Real} = Float64[],   # P
	        Θ   ::Vector{<:Real} = Float64[],   # Q
	        s   ::Int             = 1,
	        y0  ::Real            = 1.0,
	        ε0  ::Real            = 1.0)
	
	    p, q, P, Q = length(φ), length(θq), length(Φ), length(Θ)
	    maxlag = max(p, q, P*s, Q*s, 1)
	
	    y = zeros(Float64, N + maxlag)
	    ε = zeros(Float64, N + maxlag)
	
	    # Single non‑zero impulses to start the recursion
	    y[maxlag] = y0
	    ε[maxlag] = ε0
	
	    @inbounds for t in maxlag+1 : N + maxlag
	        ar = 0.0
	        @simd for i in 1:p ; ar +=     φ[i] * y[t-i]     ; end
	        @simd for i in 1:P ; ar +=     Φ[i] * y[t-i*s]   ; end
	
	        ma = 0.0
	        @simd for j in 1:q ; ma +=     θq[j] * ε[t-j]    ; end
	        @simd for j in 1:Q ; ma +=     Θ[j] * ε[t-j*s]   ; end
	
	        y[t] = ar + ma           # ε[t] is zero for t ≥ maxlag+1
	    end
	
	    return y[maxlag+1 : maxlag+N]   # drop the padded prefix
	end
# time: 2025-07-14 21:03:59 UTC
# mode: julia
	true_p, true_q, true_P, true_Q, true_s = 1, 1, 1, 1, 4
# time: 2025-07-14 21:03:59 UTC
# mode: julia
	y = generate_sarima_deterministic(2_000;
	        φ  = [0.7], θq = [0.5],
	        Φ  = [0.4], Θ  = [0.3],
	        s  = true_s)
# time: 2025-07-14 21:03:59 UTC
# mode: julia
	train  = view(y, 1:1995)
# time: 2025-07-14 21:04:00 UTC
# mode: julia
	future = view(y, 1996:2000)
# time: 2025-07-14 21:04:34 UTC
# mode: julia
	model = fit_arima_(;
	    y              = collect(train),   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = true,            # CSS first
	    loss           = css_arima_loss_, # <-- your fast loss
	    optimizer      = lbfgs)
# time: 2025-07-14 21:04:48 UTC
# mode: julia
	model = fit_arima_(;
	    y              = collect(train),   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = true)
# time: 2025-07-14 21:05:27 UTC
# mode: julia
	model = ARIMAModel(
	    y              = collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = true)
# time: 2025-07-14 21:05:50 UTC
# mode: julia
	model = ARIMAModel(collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = true)
# time: 2025-07-14 21:05:58 UTC
# mode: julia
	model = ARIMAModel(collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = true)
# time: 2025-07-14 21:06:10 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:06:19 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:06:21 UTC
# mode: julia
	model.2
# time: 2025-07-14 21:06:23 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:06:30 UTC
# mode: julia
	model.P
# time: 2025-07-14 21:06:33 UTC
# mode: julia
	model.Q
# time: 2025-07-14 21:07:04 UTC
# mode: julia
	model = fit_arima_(;
	    y              = collect(train),   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = true,            # CSS first
	    loss           = css_arima_loss_,
# time: 2025-07-14 21:07:13 UTC
# mode: julia
	yhat = forecast(model.best_model, 5)
# time: 2025-07-14 21:07:22 UTC
# mode: julia
	yhat = predict(model.best_model, 5)
# time: 2025-07-14 21:07:36 UTC
# mode: julia
	yhat = predict(model, 5)
# time: 2025-07-14 21:07:49 UTC
# mode: julia
	yhat = predict(model; 5)
# time: 2025-07-14 21:07:55 UTC
# mode: julia
	yhat = predict(model; h=5)
# time: 2025-07-14 21:08:14 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:09:11 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:09:14 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:09:40 UTC
# mode: julia
	model = fit_arima_(;
	    y              = collect(train),   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:10:01 UTC
# mode: julia
	model = ARIMAModel(
	    y              = collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:10:15 UTC
# mode: julia
	model = ARIMAModel(
	    y              = train;   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, m = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:10:33 UTC
# mode: julia
	model = ARIMAModel(
	    train;   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:10:39 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:11:06 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:11:07 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:13:36 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-14 21:13:38 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:13:44 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:13:46 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:13:48 UTC
# mode: julia
	model.P
# time: 2025-07-14 21:13:51 UTC
# mode: julia
	model.Q
# time: 2025-07-14 21:14:08 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:14:10 UTC
# mode: julia
	model.Q
# time: 2025-07-14 21:14:12 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:15:07 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = truwe)
# time: 2025-07-14 21:15:09 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = true)
# time: 2025-07-14 21:15:12 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:16:05 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,seasonal = false
	    approximation  = false,            # CSS first
	    stepwise = true)
# time: 2025-07-14 21:16:15 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,seasonal = false,
	    approximation  = false,            # CSS first
	    stepwise = true)
# time: 2025-07-14 21:16:17 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:16:24 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:16:26 UTC
# mode: julia
	model.Q
# time: 2025-07-14 21:16:28 UTC
# mode: julia
	model.P
# time: 2025-07-14 21:16:41 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,seasonal = false,
	    approximation  = true,            # CSS first
	    stepwise = true)
# time: 2025-07-14 21:16:42 UTC
# mode: julia
	model.P
# time: 2025-07-14 21:16:46 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:16:48 UTC
# mode: julia
	model.P
# time: 2025-07-14 21:17:39 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:17:41 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:18:27 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false)
# time: 2025-07-14 21:18:55 UTC
# mode: julia
	opt = Optim.NelderMead()
# time: 2025-07-14 21:19:46 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,            # CSS first
	    stepwise = false,optimizer = opt)
# time: 2025-07-14 21:21:12 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,optimizer= () -> opt          # CSS first
	    stepwise = false)
# time: 2025-07-14 21:21:17 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,optimizer= () -> opt,          # CSS first
	    stepwise = false)
# time: 2025-07-14 21:21:21 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:22:02 UTC
# mode: julia
	using Pkg
# time: 2025-07-14 21:22:04 UTC
# mode: julia
	Pkg.activate(".")
# time: 2025-07-14 21:22:05 UTC
# mode: julia
	include("src/Pythia.jl")
# time: 2025-07-14 21:22:09 UTC
# mode: julia
	using .Pythia
# time: 2025-07-14 21:22:09 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-14 21:22:37 UTC
# mode: julia
	"""
	    generate_sarima_deterministic(N;
	        φ   = Float64[],         # length p   (non‑seasonal AR)
	        θq  = Float64[],         # length q   (non‑seasonal MA)
	        Φ   = Float64[],         # length P   (seasonal AR)
	        Θ   = Float64[],         # length Q   (seasonal MA)
	        s   = 1,                 # seasonal period
	        y0  = 1.0,               # initial y "kick"
	        ε0  = 1.0)               # initial ε "kick"
	
	Return a vector `y[1:N]` that follows the SARIMA recursion with
	**εₜ = 0 for all t ≥ maxlag+1**.  Works for any (p,q,P,Q,s) ≥ 0.
	"""
	function generate_sarima_deterministic(N;
	        φ   ::Vector{<:Real} = Float64[],   # p
	        θq  ::Vector{<:Real} = Float64[],   # q
	        Φ   ::Vector{<:Real} = Float64[],   # P
	        Θ   ::Vector{<:Real} = Float64[],   # Q
	        s   ::Int             = 1,
	        y0  ::Real            = 1.0,
	        ε0  ::Real            = 1.0)
	
	    p, q, P, Q = length(φ), length(θq), length(Φ), length(Θ)
	    maxlag = max(p, q, P*s, Q*s, 1)
	
	    y = zeros(Float64, N + maxlag)
	    ε = zeros(Float64, N + maxlag)
	
	    # Single non‑zero impulses to start the recursion
	    y[maxlag] = y0
	    ε[maxlag] = ε0
	
	    @inbounds for t in maxlag+1 : N + maxlag
	        ar = 0.0
	        @simd for i in 1:p ; ar +=     φ[i] * y[t-i]     ; end
	        @simd for i in 1:P ; ar +=     Φ[i] * y[t-i*s]   ; end
	
	        ma = 0.0
	        @simd for j in 1:q ; ma +=     θq[j] * ε[t-j]    ; end
	        @simd for j in 1:Q ; ma +=     Θ[j] * ε[t-j*s]   ; end
	
	        y[t] = ar + ma           # ε[t] is zero for t ≥ maxlag+1
	    end
	
	    return y[maxlag+1 : maxlag+N]   # drop the padded prefix
	end
# time: 2025-07-14 21:22:40 UTC
# mode: julia
	true_p, true_q, true_P, true_Q, true_s = 1, 1, 1, 1, 4
# time: 2025-07-14 21:22:40 UTC
# mode: julia
	y = generate_sarima_deterministic(2_000;
	        φ  = [0.7], θq = [0.5],
	        Φ  = [0.4], Θ  = [0.3],
	        s  = true_s)
# time: 2025-07-14 21:22:41 UTC
# mode: julia
	train  = view(y, 1:1995)
# time: 2025-07-14 21:22:41 UTC
# mode: julia
	future = view(y, 1996:2000)
# time: 2025-07-14 21:23:17 UTC
# mode: julia
	my_opt() = Optim.LBFGS(; m = 5, g_tol = 1e-10)
# time: 2025-07-14 21:23:31 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,optimizer= my_opt,          # CSS first
	    stepwise = false)
# time: 2025-07-14 21:23:39 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:24:23 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,optimizer = () -> Optim.NelderMead(),          # CSS first
	    stepwise = false)
# time: 2025-07-14 21:24:24 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:24:31 UTC
# mode: julia
	model = ARIMAModel(
	    collect(train);   # make sure it’s a Vector
	    d              = 0, D = 0,        # we know it’s stationary
	    max_p = 5, max_q = 5,
	    max_P = 2, max_Q = 2, s = true_s,
	    approximation  = false,optimizer = () -> Optim.NelderMead(),          # CSS first
	    stepwise = true)
# time: 2025-07-14 21:24:32 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:24:37 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:24:40 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:24:43 UTC
# mode: julia
	model.P
# time: 2025-07-14 21:24:46 UTC
# mode: julia
	model.Q
# time: 2025-07-14 21:25:02 UTC
# mode: julia
	predict(model)
# time: 2025-07-14 21:25:33 UTC
# mode: julia
	future
# time: 2025-07-14 21:25:48 UTC
# mode: julia
	predict(model)[-10:]
# time: 2025-07-14 21:26:07 UTC
# mode: julia
	predict(model)[1996:2000]
# time: 2025-07-14 21:26:19 UTC
# mode: julia
	predict(model)[1990:2000]
# time: 2025-07-14 21:26:47 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:31:25 UTC
# mode: julia
	y = generate_sarima_deterministic(300;
	        φ  = [0.7], θq = [0.5],
	        Φ  = [0.4], Θ  = [0.3],
	        s  = true_s)
# time: 2025-07-14 21:31:41 UTC
# mode: julia
	y = generate_sarima_deterministic(2_000;
	        φ  = [0.7], θq = [0.5],
	        Φ  = [0.4], Θ  = [0.3],
	        s  = true_s)
# time: 2025-07-14 21:31:48 UTC
# mode: julia
	y = generate_sarima_deterministic(2000;
	        φ  = [0.7], θq = [0.5],
	        Φ  = [0.4], Θ  = [0.3],
	        s  = true_s)
# time: 2025-07-14 21:31:55 UTC
# mode: julia
	y = generate_sarima_deterministic(200;
	        φ  = [0.7], θq = [0.5],
	        Φ  = [0.4], Θ  = [0.3],
	        s  = true_s)
# time: 2025-07-14 21:33:28 UTC
# mode: julia
	model = ARIMAModel(y;
	    h = 1,
	    p = 5, d = 0, q = 2,
	    P = 0, D = 0, Q = 0,
	    s = 1,
	    seasonal = false,
	    stationary = true,
	)
# time: 2025-07-14 21:35:59 UTC
# mode: julia
	using Pkg
# time: 2025-07-14 21:36:01 UTC
# mode: julia
	Pkg.activate(".")
# time: 2025-07-14 21:36:02 UTC
# mode: julia
	include("src/Pythia.jl")
# time: 2025-07-14 21:36:05 UTC
# mode: julia
	using .Pythia
# time: 2025-07-14 21:36:05 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-14 21:36:37 UTC
# mode: julia
	using Random
	
	# 1.  Generator ------------------------------------------------------------
# time: 2025-07-14 21:36:37 UTC
# mode: julia
	function generate_arma11(n;
	        φ   = 0.7,
	        θ   = 0.5,
	        σ   = 1.0,
	        seed = 20250714,
	        burn = 300)
	
	    Random.seed!(seed)
	    ε = randn(n + burn) .* σ
	    y = zeros(n + burn)
	    for t in 2:(n + burn)
	        y[t] = φ*y[t-1] + ε[t] + θ*ε[t-1]
	    end
	    return y[(burn+1):end]
	end
	
	# 2.  Create 3 000 observations, keep only the first 2 995 ---------------
# time: 2025-07-14 21:36:37 UTC
# mode: julia
	y_full   = generate_arma11(3_000)
# time: 2025-07-14 21:36:37 UTC
# mode: julia
	y_train  = view(y_full, 1:2_995)   # avoids copying; use collect(y_train) if needed
	
	# 3.  Build the ARIMAModel with *your* keywords --------------------------
# time: 2025-07-14 21:36:38 UTC
# mode: julia
	model = ARIMAModel(y_train;
	    h            = 1,
	    seasonal     = false,
	    stationary   = true,
	    approximation = true,
	    stepwise     = true,
	    trace        = true,
	)
	
	# 4.  Fit it --------------------------------------------------------------
# time: 2025-07-14 21:36:38 UTC
# mode: julia
	fit_arima_(model)   # or whatever your auto‑fit wrapper is called
# time: 2025-07-14 21:36:47 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:36:49 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:37:08 UTC
# mode: julia
	using Pkg
# time: 2025-07-14 21:37:08 UTC
# mode: julia
	Pkg.activate(".")
# time: 2025-07-14 21:37:09 UTC
# mode: julia
	include("src/Pythia.jl")
# time: 2025-07-14 21:37:12 UTC
# mode: julia
	using .Pythia
# time: 2025-07-14 21:37:12 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-14 21:37:23 UTC
# mode: julia
	function generate_arma11(n;
	        φ   = 0.7,
	        θ   = 0.5,
	        σ   = 1.0,
	        seed = 20250714,
	        burn = 300)
	
	    Random.seed!(seed)
	    ε = randn(n + burn) .* σ
	    y = zeros(n + burn)
	    for t in 2:(n + burn)
	        y[t] = φ*y[t-1] + ε[t] + θ*ε[t-1]
	    end
	    return y[(burn+1):end]
	end
	
	# 2.  Create 3 000 observations, keep only the first 2 995 ---------------
# time: 2025-07-14 21:37:23 UTC
# mode: julia
	y_full   = generate_arma11(3_000)
# time: 2025-07-14 21:37:32 UTC
# mode: julia
	using Random
# time: 2025-07-14 21:37:37 UTC
# mode: julia
	function generate_arma11(n;
	        φ   = 0.7,
	        θ   = 0.5,
	        σ   = 1.0,
	        seed = 20250714,
	        burn = 300)
	
	    Random.seed!(seed)
	    ε = randn(n + burn) .* σ
	    y = zeros(n + burn)
	    for t in 2:(n + burn)
	        y[t] = φ*y[t-1] + ε[t] + θ*ε[t-1]
	    end
	    return y[(burn+1):end]
	end
	
	# 2.  Create 3 000 observations, keep only the first 2 995 ---------------
# time: 2025-07-14 21:37:37 UTC
# mode: julia
	y_full   = generate_arma11(3_000)
# time: 2025-07-14 21:37:38 UTC
# mode: julia
	y_train  = view(y_full, 1:2_995)
# time: 2025-07-14 21:37:59 UTC
# mode: julia
	model = ARIMAModel(y_train;
	    h            = 1,
	    seasonal     = false,
	    stationary   = true,
	    approximation = false,
	    stepwise     = false,
	    trace        = true,
	)
# time: 2025-07-14 21:38:12 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:38:25 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:38:28 UTC
# mode: julia
	model.
	q
# time: 2025-07-14 21:38:37 UTC
# mode: julia
	model = ARIMAModel(y_train;
	    h            = 1,
	    seasonal     = false,
	    stationary   = true,
	    approximation = false,
	    stepwise     = true,
	    trace        = true,
	)
# time: 2025-07-14 21:38:40 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:38:42 UTC
# mode: julia
	model = ARIMAModel(y_train;
	    h            = 1,
	    seasonal     = false,
	    stationary   = true,
	    approximation = false,
	    stepwise     = true,
	    trace        = true,
	)
# time: 2025-07-14 21:38:45 UTC
# mode: julia
	model.
	q
# time: 2025-07-14 21:38:50 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:38:53 UTC
# mode: julia
	model.
	q
# time: 2025-07-14 21:38:57 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:39:15 UTC
# mode: julia
	model = ARIMAModel(y_train;
	    h            = 1,
	    seasonal     = false,
	    stationary   = true,
	    approximation = true,
	    stepwise     = true,
	    trace        = true,
	)
# time: 2025-07-14 21:39:18 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:39:21 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:39:23 UTC
# mode: julia
	model.q
# time: 2025-07-14 21:43:06 UTC
# mode: julia
	function generate_arma11(n;
	                φ   = 0.7,
	                θ   = 0.5,
	                σ   = 1.0,
	                seed = 1234,
	                burn = 500)
	        
	            Random.seed!(seed)
	            ε = randn(n + burn) .* σ
	            y = zeros(n + burn)
	            for t in 2:(n + burn)
	                y[t] = φ * y[t-1] + ε[t] + θ * ε[t-1]
	            end
	            return y[(burn+1):end]
	        end
# time: 2025-07-14 21:43:15 UTC
# mode: julia
	# t
# time: 2025-07-14 21:43:20 UTC
# mode: julia
	y = generate_arma11(5000);
# time: 2025-07-14 21:44:07 UTC
# mode: julia
	function generate_arma11(n;
	                φ   = 0.7,
	                θ   = 0.5,
	                σ   = 1.0,
	                seed = 20250714,   # reproducible
	                burn = 300)        # drop startup transients
	        
	            Random.seed!(seed)
	            ε = randn(n + burn) .* σ
	            y = zeros(n + burn)
	            for t in 2:(n + burn)
	                y[t] = φ*y[t-1] + ε[t] + θ*ε[t-1]
	            end
	            return y[(burn+1):end]
	        end
# time: 2025-07-14 21:44:10 UTC
# mode: julia
	y  = generate_arma11(3000)
# time: 2025-07-14 21:44:36 UTC
# mode: julia
	model = ARIMAModel(y;
	            h = 1,
	            seasonal = false,
	            stationary = true,approximation = true,stepwise = true,trace=true
	        )
# time: 2025-07-14 21:44:42 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-14 21:44:44 UTC
# mode: julia
	mfit_arima_manual_(model)
# time: 2025-07-14 21:44:51 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:44:55 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:45:13 UTC
# mode: julia
	y  = generate_arma11(5000)
	# time: 2025-07-14 20:56:11 UTC
	# mode: julia
# time: 2025-07-14 21:45:13 UTC
# mode: julia
	model = ARIMAModel(y;
	            h = 1,
	            seasonal = false,
	            stationary = true,approximation = true,stepwise = false,trace=true
	        )
# time: 2025-07-14 21:45:21 UTC
# mode: julia
	fit_arima_(model)
# time: 2025-07-14 21:45:28 UTC
# mode: julia
	model.p
# time: 2025-07-14 21:45:30 UTC
# mode: julia
	model.q
# time: 2025-07-16 19:19:02 UTC
# mode: julia
	using Pkg
# time: 2025-07-16 19:19:04 UTC
# mode: julia
	Pkg.activate(".")
# time: 2025-07-16 19:19:05 UTC
# mode: julia
	include("src/Pythia.jl")
# time: 2025-07-16 19:19:10 UTC
# mode: julia
	using .Pythia
# time: 2025-07-16 19:19:23 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-16 19:21:04 UTC
# mode: julia
	using Random, Statistics, Optim
	
	# --- Simulate ARMA(2,1) without noise ---
# time: 2025-07-16 19:21:04 UTC
# mode: julia
	function simulate_arma(y0, φ, θ, n)
	    p, q = length(φ), length(θ)
	    y = zeros(n)
	    e = zeros(n)  # no noise
	
	    for t in 1:n
	        yt = 0.0
	        for i in 1:min(t-1, p)
	            yt += φ[i] * y[t - i]
	        end
	        for j in 1:min(t-1, q)
	            yt += θ[j] * e[t - j]
	        end
	        y[t] = yt  # since e[t] = 0
	    end
	    return y
	end
	
	# --- True parameters ---
# time: 2025-07-16 19:21:04 UTC
# mode: julia
	φ_true = [0.75, -0.25]  # AR(2)
# time: 2025-07-16 19:21:05 UTC
# mode: julia
	θ_true = [0.5]          # MA(1)
# time: 2025-07-16 19:21:05 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:21:05 UTC
# mode: julia
	y = simulate_arma([0.0, 0.0], φ_true, θ_true, n)
	
	# --- Take first 1995 values for training ---
# time: 2025-07-16 19:21:05 UTC
# mode: julia
	y_train = y[1:1995]
# time: 2025-07-16 19:21:45 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:21:56 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
	
	# --- Take first 1995 values for training ---
# time: 2025-07-16 19:22:06 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = zeros(n)  # still noise-free
	
	    for t in (maximum(p, q) + 1):n
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
	
	# Use non-zero starting values
# time: 2025-07-16 19:22:06 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:22:06 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:22:57 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = zeros(n)  # no noise for now
	
	    for t in (max(p, q) + 1):n
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:23:00 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:23:00 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:23:00 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:23:00 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:23:00 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:23:01 UTC
# mode: julia
	y_train = y[1:1995]  # Use this to fit your model
# time: 2025-07-16 19:23:15 UTC
# mode: julia
	y[800:900]
# time: 2025-07-16 19:24:10 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; σ::Float64 = 1e-3)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:24:13 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:24:13 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:24:13 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:24:13 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:24:13 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:24:14 UTC
# mode: julia
	y_train = y[1:1995]
# time: 2025-07-16 19:24:30 UTC
# mode: julia
	function simulate_arma_intercept(y0, φ, θ, n; σ=0.0, μ=5.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:24:32 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:24:41 UTC
# mode: julia
	function simulate_arma_intercept(y0, φ, θ, n; σ=0.0, μ=5.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:24:54 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:25:21 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; σ::Float64 = 1e-3)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:25:35 UTC
# mode: julia
	n=200
# time: 2025-07-16 19:25:36 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; σ::Float64 = 1e-3)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:25:38 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:25:49 UTC
# mode: julia
	y[190:200]
# time: 2025-07-16 19:26:05 UTC
# mode: julia
	n=100
# time: 2025-07-16 19:26:07 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:26:26 UTC
# mode: julia
	n=5
# time: 2025-07-16 19:26:27 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:26:40 UTC
# mode: julia
	function simulate_arma(y0, φ, θ, n; σ=1.0, μ=0.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(randn(length(y0)) * σ, randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	φ_true = [0.75, -0.25]   # AR(2)
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	θ_true = [0.5]           # MA(1)
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	μ = 10.0
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	y0 = [μ, μ]              # Start from steady state
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:26:43 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; σ=σ, μ=μ)
# time: 2025-07-16 19:26:44 UTC
# mode: julia
	y_train = y[1:1995]
# time: 2025-07-16 19:27:00 UTC
# mode: julia
	function simulate_arma(y0, φ, θ, n; σ=1.0, μ=0.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(randn(length(y0)) * σ, randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:27:06 UTC
# mode: julia
	φ_true = [0.75, -0.25]   # AR(2)
# time: 2025-07-16 19:27:06 UTC
# mode: julia
	θ_true = [0.5]           # MA(1)
# time: 2025-07-16 19:27:06 UTC
# mode: julia
	μ = 10.0
# time: 2025-07-16 19:27:06 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:27:06 UTC
# mode: julia
	y0 = [μ, μ]              # Start from steady state
# time: 2025-07-16 19:27:06 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:27:10 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; σ=σ, μ=μ)
# time: 2025-07-16 19:27:27 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; σ, μ)
# time: 2025-07-16 19:28:02 UTC
# mode: julia
	function simulate_arma(y0, φ, θ, n; σ=1.0, μ=0.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(randn(length(y0)) * σ, randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	μ = 10.0
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	y0 = [μ, μ]
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; σ=σ, μ=μ)
# time: 2025-07-16 19:28:06 UTC
# mode: julia
	y_train = y[1:1995]
# time: 2025-07-16 19:28:46 UTC
# mode: julia
	function simulate_arma(y0, φ, θ, n; σ=1.0, μ=0.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(randn(length(y0)) * σ, randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	φ_true = [0.75, -0.25]   # AR(2)
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	θ_true = [0.5]           # MA(1)
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	μ = 10.0
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	y0 = [μ, μ]              # Start from steady state
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; σ=σ, μ=μ)
# time: 2025-07-16 19:28:53 UTC
# mode: julia
	y_train = y[1:1995]
# time: 2025-07-16 19:29:01 UTC
# mode: julia
	function simulate_arma_intercept(y0, φ, θ, n; σ=0.0, μ=5.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:29:13 UTC
# mode: julia
	function simulate_arma_intercept(y0, φ, θ, n; σ=2.0, μ=5.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(zeros(length(y0)), randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        y[t] += μ
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:29:24 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n;)
# time: 2025-07-16 19:30:03 UTC
# mode: julia
	n = 5
# time: 2025-07-16 19:30:04 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n;)
# time: 2025-07-16 19:30:38 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; σ=1.0, μ=0.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = vcat(randn(length(y0)) * σ, randn(n - length(y0)) * σ)
	
	    for t in (max(p, q) + 1):n
	        yt = μ
	        for i in 1:p
	            yt += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            yt += θ[j] * e[t - j]
	        end
	        y[t] = yt + e[t]  # ← Important: Add current noise
	    end
	    return y
	end
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	μ = 10.0
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	y0 = [μ, μ]
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	n = 5
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; σ=σ, μ=μ)
# time: 2025-07-16 19:30:44 UTC
# mode: julia
	println(y)
# time: 2025-07-16 19:30:53 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, 200; σ=σ, μ=μ)
# time: 2025-07-16 19:32:59 UTC
# mode: julia
	model = ARIMAModel(stationary_y, p=1, d=result.d, q=1, P=1, D=result.D, Q=1, s=12)
# time: 2025-07-16 19:33:01 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-16 19:33:10 UTC
# mode: julia
	model = ARIMAModel(stationary_y, p=1, d=0, q=1, P=1, D=0, Q=1, s=12)
# time: 2025-07-16 19:33:17 UTC
# mode: julia
	model = ARIMAModel(y, p=1, d=0, q=1, P=1, D=0, Q=1, s=12)
# time: 2025-07-16 19:33:29 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-16 19:33:55 UTC
# mode: julia
	model = fit_arima_manual_(model)
# time: 2025-07-16 19:33:59 UTC
# mode: julia
	predict(model)
# time: 2025-07-16 19:38:13 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = zeros(n)  # no noise for now
	
	    for t in (max(p, q) + 1):n
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	    end
	    return y
	end
# time: 2025-07-16 19:38:17 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:38:17 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:38:17 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:38:17 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:38:17 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n)
# time: 2025-07-16 19:38:18 UTC
# mode: julia
	y_train = y[1:1995]  # Use this to fit your model
# time: 2025-07-16 19:38:25 UTC
# mode: julia
	y_train = y[1:995]  # Use this to fit your model
# time: 2025-07-16 19:38:34 UTC
# mode: julia
	y_train = y[1:25]  # Use this to fit your model
# time: 2025-07-16 19:38:39 UTC
# mode: julia
	y_train = y[1:2]  # Use this to fit your model
# time: 2025-07-16 19:38:44 UTC
# mode: julia
	y_train = y[1:5]  # Use this to fit your model
# time: 2025-07-16 19:38:50 UTC
# mode: julia
	y_train = y[1:10]  # Use this to fit your model
# time: 2025-07-16 19:39:33 UTC
# mode: julia
	model = ARIMAModel(y_train, p=2, d=0, q=1, P=1, D=0, Q=1, s=12)
# time: 2025-07-16 19:39:41 UTC
# mode: julia
	fit_arima_manual_(model)
# time: 2025-07-16 19:39:50 UTC
# mode: julia
	model = fit_arima_manual_(model)
# time: 2025-07-16 19:39:55 UTC
# mode: julia
	predict(model)
# time: 2025-07-16 19:40:02 UTC
# mode: julia
	y[5:10]
# time: 2025-07-16 19:40:25 UTC
# mode: julia
	yp=predict(model)
# time: 2025-07-16 19:40:29 UTC
# mode: julia
	yp[5:10]
# time: 2025-07-16 19:42:45 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; μ::Float64 = 0.0, σ::Float64 = 1.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = randn(n) .* σ  # Gaussian noise with standard deviation σ
	
	    for t in (max(p, q) + 1):n
	        y[t] = μ  # Start with mean
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	        y[t] += e[t]  # Current noise term
	    end
	    return y
	end
# time: 2025-07-16 19:43:19 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; μ::Float64 = 0.0, σ::Float64 = 1.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = randn(n) .* σ  # Gaussian noise
	
	    for t in (max(p, q) + 1):n
	        y[t] = μ  # Start with mean
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	        y[t] += e[t]  # Add current noise term
	    end
	    return y
	end
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	μ = 2.0
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:43:23 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; μ=μ, σ=σ)
# time: 2025-07-16 19:43:31 UTC
# mode: julia
	using Plots, Statistics
# time: 2025-07-16 19:49:32 UTC
# mode: julia
	using Pkg
# time: 2025-07-16 19:49:34 UTC
# mode: julia
	Pkg.activate(".")
# time: 2025-07-16 19:49:35 UTC
# mode: julia
	include("src/Pythia.jl")
# time: 2025-07-16 19:49:41 UTC
# mode: julia
	using .Pythia
# time: 2025-07-16 19:49:41 UTC
# mode: julia
	include("src/algorithms/arma.jl")
# time: 2025-07-16 19:49:54 UTC
# mode: julia
	function simulate_arma(y0::Vector{Float64}, φ::Vector{Float64}, θ::Vector{Float64}, n::Int; μ::Float64 = 0.0, σ::Float64 = 1.0)
	    p, q = length(φ), length(θ)
	    y = vcat(y0, zeros(n - length(y0)))
	    e = randn(n) .* σ  # Gaussian noise
	
	    for t in (max(p, q) + 1):n
	        y[t] = μ  # Start with mean
	        for i in 1:p
	            y[t] += φ[i] * y[t - i]
	        end
	        for j in 1:q
	            y[t] += θ[j] * e[t - j]
	        end
	        y[t] += e[t]  # Add current noise term
	    end
	    return y
	end
# time: 2025-07-16 19:49:58 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:49:59 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:49:59 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:49:59 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:49:59 UTC
# mode: julia
	μ = 2.0
# time: 2025-07-16 19:49:59 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:49:59 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; μ=μ, σ=σ)
# time: 2025-07-16 19:50:30 UTC
# mode: julia
	model = fit_arima_manual_(model)
# time: 2025-07-16 19:51:19 UTC
# mode: julia
	y_train = y[1:995]
# time: 2025-07-16 19:51:58 UTC
# mode: julia
	model = ARIMAModel(y_train, p=2, d=0, q=1, P=1, D=0, Q=1, s=12)
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	μ = 2.0
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:52:03 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; μ=μ, σ=σ)
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	φ_true = [0.75, -0.25]
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	θ_true = [0.5]
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	y0 = [1.0, -1.0]
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	n = 2000
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	μ = 2.0
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	σ = 1.0
# time: 2025-07-16 19:52:08 UTC
# mode: julia
	y = simulate_arma(y0, φ_true, θ_true, n; μ=μ, σ=σ)
# time: 2025-07-16 19:53:13 UTC
# mode: julia
	model = ARIMAModel(y_train, p=1, d=0, q=1, P=0,D=0, Q=0,s=12)
# time: 2025-07-16 19:53:27 UTC
# mode: julia
	model = fit_arima_manual_(model)
# time: 2025-07-16 19:53:46 UTC
# mode: julia
	predict(model)
# time: 2025-07-16 19:54:25 UTC
# mode: julia
	y_train[1:35]
# time: 2025-07-16 19:54:30 UTC
# mode: julia
	model = ARIMAModel(y_train, p=1, d=0, q=1, P=0,D=0, Q=0,s=12)
# time: 2025-07-16 19:54:32 UTC
# mode: julia
	model = fit_arima_manual_(model)
# time: 2025-07-16 19:54:38 UTC
# mode: julia
	predict(model)
# time: 2025-07-16 19:54:50 UTC
# mode: julia
	y_train[35:30]
# time: 2025-07-16 19:54:56 UTC
# mode: julia
	y_train[35:40]
