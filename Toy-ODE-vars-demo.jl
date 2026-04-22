using DifferentialEquations
using QuasiMonteCarlo
using Base.Threads
using SurrogatesPolyChaos
using Surrogates
using LinearAlgebra
using Statistics

BLAS.set_num_threads(Threads.nthreads())

# ------------------------------------------------------------
# Demo script: global sensitivity analysis for a Lotka-Volterra
# predator-prey model using Sobol/PCE and a simplified
# VARS-style workflow.
# This is an illustrative educational example.
# ------------------------------------------------------------

# =====================================
# 1) Lotka-Volterra predator-prey ODE
# =====================================
function lotka!(du, u, p, t)
    α, β, δ, γ = p
    x, y = u
    du[1] = α * x - β * x * y
    du[2] = δ * x * y - γ * y
end

# Scalar model output: maximum prey population
function model_output(p)
    prob = ODEProblem(lotka!, [10.0, 5.0], (0.0, 20.0), p)
    sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-6)
    return maximum(sol[1, :])
end

# =====================================
# 2) Parameter domain and helpers
# =====================================
d = 4
lb = [0.5, 0.1, 0.1, 0.5]
ub = [1.5, 0.5, 0.5, 1.5]

function scale_unit_to_bounds(u::AbstractVector{<:Real}, lb::AbstractVector, ub::AbstractVector)
    return lb .+ (ub .- lb) .* u
end

function scale_unit_to_bounds(M::AbstractMatrix{<:Real}, lb::AbstractVector, ub::AbstractVector)
    return hcat([scale_unit_to_bounds(M[:, i], lb, ub) for i in 1:size(M, 2)]...)
end

# =====================================
# 3) Sobol sampling + PCE sensitivity
# =====================================
N = 500
sampler = SobolSample(Shift())
A, B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler, 2)
X = hcat(A, B)

Y = Vector{Float64}(undef, size(X, 2))
@threads for i in 1:size(X, 2)
    Y[i] = model_output(X[:, i])
end

deg = 3
orthos = SurrogatesPolyChaos.MultiOrthoPoly(
    [SurrogatesPolyChaos.GaussOrthoPoly(deg) for _ in 1:d],
    deg
)
xpts = [Vector(X[:, i]) for i in 1:size(X, 2)]

pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(
    xpts, Y, lb, ub;
    orthopolys = orthos
)

coeffs = pce.coeff
multiidx = pce.orthopolys.ind
varY = sum(coeffs[2:end].^2)

S1 = zeros(d)
ST = zeros(d)

for k in 2:length(coeffs)
    idxs = findall(multiidx[k, :] .> 0)
    if length(idxs) == 1
        S1[idxs[1]] += coeffs[k]^2
    end
    for j in idxs
        ST[j] += coeffs[k]^2
    end
end

S1 ./= varY
ST ./= varY

S2 = zeros(d, d)
for k in 2:length(coeffs)
    idxs = findall(multiidx[k, :] .> 0)
    if length(idxs) == 2
        i, j = idxs
        S2[i, j] = coeffs[k]^2
        S2[j, i] = coeffs[k]^2
    end
end
S2 ./= varY

println("PCE — First-order indices: ", round.(S1, digits=5))
println("PCE — Total-order indices: ", round.(ST, digits=5))
println("PCE — Second-order matrix:")
println(round.(S2, digits=5))

# =====================================
# 4) Simplified VARS-style workflow
# =====================================
function vars_matrices_demo(star_centers::Int, d::Int, h::Float64)
    sampler = SobolSample(Shift())
    centers = QuasiMonteCarlo.generate_design_matrices(
        star_centers, zeros(d), ones(d), sampler, 1
    )[1]

    point_info = Tuple{Int, Int, Bool}[]
    point_vectors = Vector{Vector{Float64}}()

    for star in 1:star_centers
        center = centers[:, star]
        push!(point_vectors, center)
        push!(point_info, (star, 0, true))

        for dim in 1:d
            c_dim = center[dim]
            traj_values = filter(
                x -> x != c_dim,
                unique(vcat(c_dim % h:h:1.0, c_dim % h:-h:0.0))
            )

            for tv in traj_values
                new_point = copy(center)
                new_point[dim] = clamp(tv, 0.0, 0.9999)
                push!(point_vectors, new_point)
                push!(point_info, (star, dim, false))
            end
        end
    end

    X_unit = hcat(point_vectors...)
    return (X_unit = X_unit, info = point_info)
end

function gsa_vars_demo(Y::Vector{Float64}, info::Vector{Tuple{Int, Int, Bool}}, star_centers::Int, d::Int)
    center_idx = findall(p -> p[3] == true, info)
    if length(center_idx) < 2
        return fill(NaN, d)
    end

    VY = var(@view Y[center_idx])
    if VY < 1e-12
        return zeros(d)
    end

    Ti = zeros(d)

    for dim in 1:d
        variogram_sum = 0.0
        covariogram_sum = 0.0
        stars_with_data = 0

        for star in 1:star_centers
            traj_positions = findall(p -> p[1] == star && p[2] == dim, info)
            if length(traj_positions) < 2
                continue
            end

            p1 = Float64[]
            p2 = Float64[]

            for j in 1:length(traj_positions)-1
                push!(p1, Y[traj_positions[j]])
                push!(p2, Y[traj_positions[j + 1]])
            end

            variogram_i = 0.5 * mean((p1 .- p2).^2)
            covariogram_i = cov(p1, p2)

            if !isnan(variogram_i) && !isnan(covariogram_i)
                variogram_sum += variogram_i
                covariogram_sum += covariogram_i
                stars_with_data += 1
            end
        end

        if stars_with_data > 0
            Ti[dim] = (variogram_sum / stars_with_data + covariogram_sum / stars_with_data) / VY
        else
            Ti[dim] = NaN
        end
    end

    return Ti
end

function estimate_vars_cost(r::Int, d::Int, h::Float64)
    sampler = SobolSample(Shift())
    test_centers = QuasiMonteCarlo.generate_design_matrices(
        min(10, r), zeros(d), ones(d), sampler, 1
    )[1]

    total_points = 0
    for i in 1:size(test_centers, 2)
        center = test_centers[:, i]
        points_for_center = 1
        for dim in 1:d
            c_dim = center[dim]
            traj_values = filter(
                x -> x != c_dim,
                unique(vcat(c_dim % h:h:1.0, c_dim % h:-h:0.0))
            )
            points_for_center += length(traj_values)
        end
        total_points += points_for_center
    end

    avg_per_star = total_points / size(test_centers, 2)
    return round(Int, r * avg_per_star)
end

function find_vars_params_demo(target_cost::Int, d::Int)
    h_options = [0.1, 0.2]
    r_options = [50, 100, 200]

    best = (r = 0, h = 0.0, cost = 0, diff = Inf)

    for h_val in h_options
        for r_val in r_options
            est_cost = estimate_vars_cost(r_val, d, h_val)
            diff = abs(est_cost - target_cost)

            if diff < best.diff
                best = (r = r_val, h = h_val, cost = est_cost, diff = diff)
            end
        end
    end

    return best
end

vars_total_cost = 5000
params = find_vars_params_demo(vars_total_cost, d)
println("VARS demo params → r=$(params.r), h=$(params.h), estimated_cost=$(params.cost)")

vars_design = vars_matrices_demo(params.r, d, params.h)
X_unit = vars_design.X_unit
info_v = vars_design.info
X_phys = scale_unit_to_bounds(X_unit, lb, ub)

Y_vars = Vector{Float64}(undef, size(X_phys, 2))
@threads for i in 1:size(X_phys, 2)
    Y_vars[i] = model_output(X_phys[:, i])
end

ST_vars = gsa_vars_demo(Y_vars, info_v, params.r, d)
println("VARS-style — Total-order-like indices: ", round.(ST_vars, digits=5))
