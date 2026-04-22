using DifferentialEquations, Random, Distributions, DelimitedFiles, StatsPlots

# Define the birth-death ODE with 10^params transformation to ensure positivity
function birth_death!(du, u, p, t)
    # Parameters are in log10 scale, so transform them
    birth_rate = 10^p[1]
    death_rate = 10^p[2]
    
    # Population growth equation: dN/dt = birth_rate*N - death_rate*N
    du[1] = birth_rate*u[1] - death_rate*u[1]
end

# Set true parameter values in log10 scale
log_true_birth_rate = log10(0.5)   # Birth rate
log_true_death_rate = log10(0.2)   # Death rate
true_log_params = [log_true_birth_rate, log_true_death_rate]

# Note that when solving the ODE, we use the log-scale parameters
# but the actual rates used internally are:
# true_birth_rate = 10^log_true_birth_rate = 0.5
# true_death_rate = 10^log_true_death_rate = 0.2

# Initial condition
u0 = [1000.0]  # Initial population size

# Time span
tspan = (0.0, 100.0)

# Create the ODE problem
prob = ODEProblem(birth_death!, u0, tspan, true_log_params)

# Solve the ODE
sol = solve(prob, Tsit5(), saveat=0.1)

# Extract solution
t_points = sol.t
population = [sol[1, i] for i in 1:length(t_points)]

# Add some noise to mimic experimental data
rng = MersenneTwister(123)  # For reproducibility
noise_level = 0.05  # 5% noise
noisy_population = population .* (1 .+ noise_level .* randn(rng, length(population)))

# Save the "experimental" data to a file
exp_data = hcat(t_points, noisy_population)
writedlm("experimental_data.txt", exp_data)

