using Distributions
using LinearAlgebra
using Statistics
using StatsBase

# ------------------------------------------------------------
# Demo script: simplified ABC-SMC parameter inference
# for a toy exponential-decay model.
# This is an educational example, not a production framework.
# ------------------------------------------------------------

# ============================================================
# 1) Toy forward model
# y(t) = y0 * exp(-k*t)
# ============================================================
function simulate_decay(params, t)
    y0, k = params
    return y0 .* exp.(-k .* t)
end

# Synthetic observed data
t_obs = collect(0.0:1.0:10.0)
true_params = [5.0, 0.3]
y_obs_clean = simulate_decay(true_params, t_obs)

rng = MersenneTwister(42)
y_obs = y_obs_clean .+ 0.15 .* randn(rng, length(t_obs))

# ============================================================
# 2) Distance function
# ============================================================
function distance_function(params)
    y_sim = simulate_decay(params, t_obs)
    return mean((y_sim .- y_obs).^2)
end

# ============================================================
# 3) Prior definition
# ============================================================
priors = [
    Uniform(1.0, 10.0),   # y0
    Uniform(0.05, 1.0)    # k
]

function sample_from_priors(priors)
    return [rand(p) for p in priors]
end

function prior_pdf(priors, x)
    vals = [pdf(priors[i], x[i]) for i in eachindex(priors)]
    return prod(vals)
end

function in_support(priors, x)
    return all(pdf(priors[i], x[i]) > 0 for i in eachindex(priors))
end

# ============================================================
# 4) Weighted covariance helper
# ============================================================
function weighted_covariance(samples::Matrix{Float64}, weights_vec::Vector{Float64})
    n, d = size(samples)
    μ = vec(sum(samples .* weights_vec, dims=1))
    centered = samples .- μ'
    Σ = zeros(d, d)

    for i in 1:n
        Σ .+= weights_vec[i] .* (centered[i, :] * centered[i, :]')
    end

    return Σ
end

# ============================================================
# 5) Simplified ABC-SMC
# ============================================================
function abc_smc_demo(
    priors,
    distance_function;
    N = 200,
    n_steps = 4,
    quantile_level = 0.5,
    kernel_scale = 2.0,
    rng = Random.default_rng()
)
    d = length(priors)

    particles_history = Vector{Matrix{Float64}}()
    weights_history = Vector{Vector{Float64}}()
    epsilon_history = Float64[]

    # -----------------------------
    # Initial population
    # -----------------------------
    particles = zeros(N, d)
    distances = zeros(N)

    for i in 1:N
        θ = sample_from_priors(priors)
        particles[i, :] = θ
        distances[i] = distance_function(θ)
    end

    ε = quantile(distances, quantile_level)
    keep = distances .<= ε

    particles = particles[keep, :]
    distances = distances[keep]

    # If more than needed survived, trim to N/2
    n_keep = min(size(particles, 1), round(Int, N * quantile_level))
    particles = particles[1:n_keep, :]
    distances = distances[1:n_keep]

    weights_vec = fill(1.0 / n_keep, n_keep)

    push!(particles_history, copy(particles))
    push!(weights_history, copy(weights_vec))
    push!(epsilon_history, ε)

    println("Step 1: epsilon = $(round(ε, digits=6)), particles kept = $n_keep")

    # -----------------------------
    # Sequential populations
    # -----------------------------
    for step in 2:n_steps
        Σ = weighted_covariance(particles, weights_vec)

        # Small diagonal jitter for numerical stability
        Σ += 1e-6I
        kernel_cov = kernel_scale .* Σ
        kernel = MvNormal(zeros(d), Symmetric(kernel_cov))

        new_particles = zeros(n_keep, d)
        new_distances = zeros(n_keep)
        new_weights = zeros(n_keep)

        accepted = 0
        attempts = 0

        while accepted < n_keep
            attempts += 1

            parent_idx = sample(1:n_keep, weights(weights_vec))
            θ_parent = particles[parent_idx, :]
            θ_prop = θ_parent + rand(rng, kernel)

            if !in_support(priors, θ_prop)
                continue
            end

            dist = distance_function(θ_prop)
            if dist > ε
                continue
            end

            accepted += 1
            new_particles[accepted, :] = θ_prop
            new_distances[accepted] = dist

            numerator = prior_pdf(priors, θ_prop)
            denominator = 0.0

            for j in 1:n_keep
                denominator += weights_vec[j] * pdf(
                    MvNormal(particles[j, :], Symmetric(kernel_cov)),
                    θ_prop
                )
            end

            new_weights[accepted] = denominator > 0 ? numerator / denominator : 0.0
        end

        # Normalize weights
        if sum(new_weights) > 0
            new_weights ./= sum(new_weights)
        else
            new_weights .= 1.0 / length(new_weights)
        end

        particles = new_particles
        weights_vec = new_weights
        ε = quantile(new_distances, quantile_level)

        push!(particles_history, copy(particles))
        push!(weights_history, copy(weights_vec))
        push!(epsilon_history, ε)

        println("Step $step: epsilon = $(round(ε, digits=6)), attempts = $attempts")
    end

    return (
        particles_history = particles_history,
        weights_history = weights_history,
        epsilon_history = epsilon_history
    )
end

# ============================================================
# 6) Run demo
# ============================================================
result = abc_smc_demo(
    priors,
    distance_function;
    N = 200,
    n_steps = 5,
    quantile_level = 0.5,
    kernel_scale = 2.0,
    rng = rng
)

final_particles = result.particles_history[end]
final_weights = result.weights_history[end]

posterior_mean = vec(sum(final_particles .* final_weights, dims=1))

println("\nTrue parameters:")
println(true_params)

println("\nPosterior mean estimate:")
println(round.(posterior_mean, digits=4))

println("\nEpsilon schedule:")
println(round.(result.epsilon_history, digits=6))
