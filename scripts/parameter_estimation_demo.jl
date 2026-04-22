using DifferentialEquations
using Random
using Distributions
using DelimitedFiles
using StatsPlots

# ------------------------------------------------------------
# Simplified demo: parameter estimation for a birth-death model
# This script is a public-safe example and does not reproduce
# the full inference pipeline used in the research project.
# ------------------------------------------------------------

# Birth-death ODE
# Parameters are provided in log10 scale to keep rates positive
function birth_death!(du, u, p, t)
    birth_rate = 10.0^p[1]
    death_rate = 10.0^p[2]
    du[1] = (birth_rate - death_rate) * u[1]
end

# Load example experimental data
exp_data = readdlm("experimental_data.txt")
t_exp = exp_data[:, 1]
pop_exp = exp_data[:, 2]

# Initial condition and simulation time span
u0 = [1000.0]
tspan = (minimum(t_exp), maximum(t_exp))

# Mean squared error loss function
function loss_function(log_params, t_exp, pop_exp, u0, tspan)
    prob = ODEProblem(birth_death!, u0, tspan, log_params)
    sol = solve(prob, Tsit5(), saveat=t_exp)

    if sol.retcode != :Success
        return Inf
    end

    pop_sim = [sol[1, i] for i in 1:length(t_exp)]
    return mean((pop_sim .- pop_exp).^2)
end

# Parameter ranges in log10 scale
parameter_ranges_log10 = [
    (log10(0.25), log10(1.0)),   # birth rate
    (log10(0.1),  log10(0.4))    # death rate
]

param_names = ["Birth Rate", "Death Rate"]

# Random search settings
rng = MersenneTwister(42)
n_samples = 1000

sampled_log_params = zeros(2, n_samples)
loss_values = fill(Inf, n_samples)

best_loss = Inf
best_log_params = zeros(2)

# Random parameter search
for i in 1:n_samples
    current_log_params = [
        rand(rng, Uniform(parameter_ranges_log10[1][1], parameter_ranges_log10[1][2])),
        rand(rng, Uniform(parameter_ranges_log10[2][1], parameter_ranges_log10[2][2]))
    ]

    current_loss = loss_function(current_log_params, t_exp, pop_exp, u0, tspan)

    sampled_log_params[:, i] = current_log_params
    loss_values[i] = current_loss

    if current_loss < best_loss
        best_loss = current_loss
        best_log_params .= current_log_params
    end
end

# Convert best parameters back to linear scale
best_params = 10 .^ best_log_params

println("Best estimated birth rate: ", best_params[1])
println("Best estimated death rate: ", best_params[2])
println("Best loss: ", best_loss)

# Solve model with best-fit parameters
prob_best = ODEProblem(birth_death!, u0, tspan, best_log_params)
sol_best = solve(prob_best, Tsit5(), saveat=t_exp)
best_fit_population = [sol_best[1, i] for i in 1:length(t_exp)]

# Plot experimental data vs best-fit model
p1 = plot(t_exp, pop_exp,
    seriestype = :scatter,
    label = "Experimental data",
    xlabel = "Time",
    ylabel = "Population",
    title = "Best-Fit Model vs Experimental Data"
)

plot!(p1, t_exp, best_fit_population,
    linewidth = 2,
    label = "Best-fit simulation"
)

savefig(p1, "best_fit_vs_experimental_demo.png")

# Plot distributions of sampled parameters
linear_birth_samples = 10 .^ sampled_log_params[1, :]
linear_death_samples = 10 .^ sampled_log_params[2, :]

p2 = histogram(linear_birth_samples,
    bins = 30,
    normalize = true,
    alpha = 0.6,
    label = "Samples",
    xlabel = "Birth rate",
    ylabel = "Density",
    title = "Sampled Birth Rate Values"
)

vline!(p2, [best_params[1]], linewidth = 2, label = "Best estimate")

p3 = histogram(linear_death_samples,
    bins = 30,
    normalize = true,
    alpha = 0.6,
    label = "Samples",
    xlabel = "Death rate",
    ylabel = "Density",
    title = "Sampled Death Rate Values"
)

vline!(p3, [best_params[2]], linewidth = 2, label = "Best estimate")

combined_plot = plot(p2, p3, layout = (1, 2), size = (900, 400))
savefig(combined_plot, "parameter_sampling_demo.png")
