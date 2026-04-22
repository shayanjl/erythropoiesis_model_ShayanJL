using DifferentialEquations
using Plots
using Statistics

# ------------------------------------------------------------
# Demo script: intervention-response analysis in a generic
# compartmental model with repeated pulse inputs.
# This is a synthetic educational example.
# ------------------------------------------------------------

atol = 1e-8
rtol = 1e-8

# ============================================================
# 1) Baseline model
# States:
#   X1 = source compartment
#   X2 = intermediate compartment
#   X3 = precursor compartment
#   X4 = mature compartment
#   X5 = support compartment
#   X6 = competing population
#   S  = signaling factor
# ============================================================
function baseline_model!(du, u, p, t)
    growth1, growth2, growth3, growth4, growth5,
    death1, death2, death3, death4, death5,
    trans12, trans23, trans34,
    support_effect2, support_effect3,
    comp_growth, comp_death,
    signal_prod, signal_deg, signal_effect,
    K = p

    X1, X2, X3, X4, X5, X6, S = u

    total_cells = X1 + X2 + X3 + X4 + X5 + X6
    competition = max(0.0, 1.0 - total_cells / K)

    du[1] = growth1 * X1 * competition - death1 * X1 - trans12 * X1
    du[2] = trans12 * X1 + (growth2 + support_effect2 * X5) * X2 * competition - death2 * X2 - trans23 * X2 / (1 + signal_effect * S)
    du[3] = trans23 * X2 / (1 + signal_effect * S) + (growth3 + support_effect3 * X5) * X3 * competition - death3 * X3 - trans34 * X3
    du[4] = trans34 * X3 + growth4 * X4 * competition - death4 * X4
    du[5] = growth5 * X5 * competition - death5 * X5
    du[6] = comp_growth * X6 * competition - comp_death * X6
    du[7] = signal_prod * X6 - signal_deg * S

    return nothing
end

# ============================================================
# 2) Intervention model with repeated pulses
# ============================================================
function intervention_model!(du, u, p, t)
    growth1, growth2, growth3, growth4, growth5,
    death1, death2, death3, death4, death5,
    trans12, trans23, trans34,
    support_effect2, support_effect3,
    comp_growth, comp_death,
    signal_prod, signal_deg, signal_effect,
    pulse_amp, K = p

    X1, X2, X3, X4, X5, X6, S = u

    pulse_times = [48.0, 96.0, 144.0]
    pulse_duration = 2.0
    pulse_width = 0.5

    pulse = 0.0
    for pt in pulse_times
        start_t = pt - pulse_duration / 2
        end_t = pt + pulse_duration / 2
        if start_t <= t <= end_t
            pulse += pulse_amp * exp(-((t - pt)^2) / (pulse_width^2))
        end
    end

    total_cells = X1 + X2 + X3 + X4 + X5 + X6
    competition = max(0.0, 1.0 - total_cells / K)

    du[1] = growth1 * X1 * competition - death1 * X1 - trans12 * X1
    du[2] = trans12 * X1 + (growth2 + support_effect2 * X5) * X2 * competition - death2 * X2 - trans23 * X2 / (1 + signal_effect * S)
    du[3] = trans23 * X2 / (1 + signal_effect * S) + (growth3 + support_effect3 * X5) * X3 * competition - death3 * X3 - trans34 * X3
    du[4] = trans34 * X3 + growth4 * X4 * competition - death4 * X4
    du[5] = growth5 * X5 * competition - death5 * X5
    du[6] = comp_growth * X6 * competition - comp_death * X6
    du[7] = signal_prod * X6 + pulse - signal_deg * S

    return nothing
end

# ============================================================
# 3) Initial conditions and baseline parameters
# ============================================================
u0 = [
    0.08,  # X1
    0.10,  # X2
    0.12,  # X3
    0.20,  # X4
    0.07,  # X5
    0.03,  # X6
    0.00   # S
]

base_params = [
    0.20,  # growth1
    0.28,  # growth2
    0.25,  # growth3
    0.10,  # growth4
    0.08,  # growth5
    0.03,  # death1
    0.04,  # death2
    0.04,  # death3
    0.03,  # death4
    0.02,  # death5
    0.10,  # trans12
    0.12,  # trans23
    0.14,  # trans34
    0.18,  # support_effect2
    0.15,  # support_effect3
    0.35,  # comp_growth
    0.05,  # comp_death
    0.15,  # signal_prod
    0.12,  # signal_deg
    0.60,  # signal_effect
    1.00   # K
]

tspan = (0.0, 7.0 * 24.0)

# ============================================================
# 4) Baseline simulation
# ============================================================
prob_baseline = ODEProblem(baseline_model!, u0, tspan, base_params)
sol_baseline = solve(prob_baseline, TRBDF2(), abstol=atol, reltol=rtol)

baseline_final = sol_baseline.u[end]

# ============================================================
# 5) Compare baseline vs one intervention setting
# ============================================================
pulse_amp = 5.0
intervention_params = vcat(base_params, pulse_amp)

prob_intervention = ODEProblem(intervention_model!, u0, tspan, intervention_params)
sol_intervention = solve(prob_intervention, TRBDF2(), abstol=atol, reltol=rtol)

intervention_final = sol_intervention.u[end]

state_names = [
    "State 1",
    "State 2",
    "State 3",
    "State 4",
    "State 5",
    "Competing Population",
    "Signal"
]

baseline_vals = [baseline_final[i] for i in 1:7]
intervention_vals = [intervention_final[i] for i in 1:7]

p1 = bar(
    1:7,
    [baseline_vals intervention_vals],
    label=["Baseline" "Intervention"],
    title="Final state comparison",
    xlabel="State",
    ylabel="Value",
    xticks=(1:7, state_names),
    xrotation=45,
    legend=:topright
)
savefig(p1, "toy_final_state_comparison.png")

# ============================================================
# 6) Time-course comparison for selected states
# ============================================================
tvals = sol_baseline.t
baseline_X4 = [u[4] for u in sol_baseline.u]
baseline_X5 = [u[5] for u in sol_baseline.u]
baseline_X6 = [u[6] for u in sol_baseline.u]
baseline_S  = [u[7] for u in sol_baseline.u]

tvals2 = sol_intervention.t
interv_X4 = [u[4] for u in sol_intervention.u]
interv_X5 = [u[5] for u in sol_intervention.u]
interv_X6 = [u[6] for u in sol_intervention.u]
interv_S  = [u[7] for u in sol_intervention.u]

p2 = plot(tvals, baseline_X4, label="State 4 baseline", linewidth=2, xlabel="Time (hours)", ylabel="Level", title="Selected trajectories")
plot!(p2, tvals2, interv_X4, label="State 4 intervention", linewidth=2)
plot!(p2, tvals, baseline_X5, label="State 5 baseline", linewidth=2, linestyle=:dash)
plot!(p2, tvals2, interv_X5, label="State 5 intervention", linewidth=2, linestyle=:dash)
savefig(p2, "toy_selected_trajectories.png")

p3 = plot(tvals, baseline_X6, label="Competing population baseline", linewidth=2, xlabel="Time (hours)", ylabel="Level", title="Competing population and signal")
plot!(p3, tvals2, interv_X6, label="Competing population intervention", linewidth=2)
plot!(p3, tvals, baseline_S, label="Signal baseline", linewidth=2, linestyle=:dash)
plot!(p3, tvals2, interv_S, label="Signal intervention", linewidth=2, linestyle=:dash)
savefig(p3, "toy_competition_signal_trajectories.png")

# ============================================================
# 7) Scan intervention amplitudes
# ============================================================
amplitudes = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
tracked_states = [4, 5, 6, 7]
tracked_labels = ["State 4", "State 5", "Competing Population", "Signal"]

normalized_values = zeros(length(tracked_states), length(amplitudes))

for (j, amp) in enumerate(amplitudes)
    current_params = vcat(base_params, amp)
    prob = ODEProblem(intervention_model!, u0, tspan, current_params)
    sol = solve(prob, TRBDF2(), abstol=atol, reltol=rtol)
    current_final = sol.u[end]

    for (i, idx) in enumerate(tracked_states)
        normalized_values[i, j] = current_final[idx] / baseline_final[idx]
    end
end

p4 = plot(
    xlabel="Intervention amplitude",
    ylabel="Normalized final value",
    xscale=:log10,
    title="Response to intervention amplitude",
    legend=:best,
    linewidth=2
)

for i in 1:length(tracked_states)
    plot!(p4, amplitudes, normalized_values[i, :], label=tracked_labels[i], marker=:circle)
end

savefig(p4, "toy_intervention_amplitude_scan.png")

# ============================================================
# 8) Plot the dosing schedule
# ============================================================
time_points = 0.0:0.05:7.0
dose_profile = zeros(length(time_points))

injection_days = [2.0, 4.0, 6.0]
pulse_duration_days = 2.0 / 24.0
pulse_width_days = 0.5 / 24.0
base_level = 1.0
amp_plot = 1.0

for day in injection_days
    for i in eachindex(time_points)
        t = time_points[i]
        if (day - pulse_duration_days / 2) <= t <= (day + pulse_duration_days / 2)
            dose_profile[i] += amp_plot * exp(-((t - day)^2) / (pulse_width_days^2))
        end
    end
end

dose_profile .+= base_level

p5 = plot(
    time_points,
    dose_profile,
    label="",
    xlabel="Time (days)",
    ylabel="Signal input level",
    title="Toy dosing schedule",
    linewidth=2
)
savefig(p5, "toy_dosing_schedule.png")

println("Saved figures:")
println(" - toy_final_state_comparison.png")
println(" - toy_selected_trajectories.png")
println(" - toy_competition_signal_trajectories.png")
println(" - toy_intervention_amplitude_scan.png")
println(" - toy_dosing_schedule.png")
