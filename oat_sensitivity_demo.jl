using DifferentialEquations
using Plots
using Statistics
using Printf
using Measures

# ------------------------------------------------------------
# Demo script: one-at-a-time sensitivity analysis for a simple
# compartmental ODE model using synthetic reference data.
# This is a generic educational example.
# ------------------------------------------------------------

# ============================================================
# 1) Simple multi-compartment ODE model
# ============================================================
function compartment_model!(du, u, p, t)
    growth_A, growth_B, growth_C,
    death_A, death_B, death_C,
    diff_AB, diff_BC,
    signal_prod, signal_deg,
    signal_effect, K = p

    A, B, C, Signal = u

    total_cells = A + B + C
    competition = max(0.0, 1.0 - total_cells / K)

    du[1] = growth_A * A * competition - death_A * A - diff_AB * A
    du[2] = diff_AB * A + growth_B * B * competition - death_B * B - diff_BC * B / (1 + signal_effect * Signal)
    du[3] = diff_BC * B / (1 + signal_effect * Signal) + growth_C * C * competition - death_C * C
    du[4] = signal_prod * C - signal_deg * Signal

    return nothing
end

# ============================================================
# 2) Generate synthetic reference data
# ============================================================
tspan = (0.0, 30.0)
save_times = [10.0, 20.0, 30.0]

u0 = [
    0.15,  # A
    0.10,  # B
    0.05,  # C
    0.00   # Signal
]

true_params = [
    0.50,  # growth_A
    0.40,  # growth_B
    0.25,  # growth_C
    0.08,  # death_A
    0.06,  # death_B
    0.04,  # death_C
    0.18,  # diff_AB
    0.15,  # diff_BC
    0.12,  # signal_prod
    0.10,  # signal_deg
    0.80,  # signal_effect
    1.00   # K
]

param_names = [
    "growth_A", "growth_B", "growth_C",
    "death_A", "death_B", "death_C",
    "diff_AB", "diff_BC",
    "signal_prod", "signal_deg",
    "signal_effect", "K"
]

prob_true = ODEProblem(compartment_model!, u0, tspan, true_params)
sol_true = solve(prob_true, Tsit5(), saveat=save_times, abstol=1e-8, reltol=1e-8)

# Synthetic reference data from states A, B, C at selected times
reference_data = hcat(
    [sol_true[1, i] for i in 1:length(save_times)],
    [sol_true[2, i] for i in 1:length(save_times)],
    [sol_true[3, i] for i in 1:length(save_times)]
)

# ============================================================
# 3) Model evaluation function
# ============================================================
function model_loss(params)
    try
        prob = ODEProblem(compartment_model!, u0, tspan, params)
        sol = solve(prob, Tsit5(), saveat=save_times, abstol=1e-8, reltol=1e-8)

        if sol.retcode != ReturnCode.Success
            return 1e6
        end

        sim_data = hcat(
            [sol[1, i] for i in 1:length(save_times)],
            [sol[2, i] for i in 1:length(save_times)],
            [sol[3, i] for i in 1:length(save_times)]
        )

        return mean((sim_data .- reference_data).^2)
    catch
        return 1e6
    end
end

# ============================================================
# 4) Parameter ranges
# ============================================================
parameter_ranges = [
    (0.2, 0.8),   # growth_A
    (0.2, 0.7),   # growth_B
    (0.1, 0.5),   # growth_C
    (0.01, 0.15), # death_A
    (0.01, 0.12), # death_B
    (0.01, 0.10), # death_C
    (0.05, 0.30), # diff_AB
    (0.05, 0.25), # diff_BC
    (0.05, 0.20), # signal_prod
    (0.05, 0.20), # signal_deg
    (0.1, 1.5),   # signal_effect
    (0.7, 1.3)    # K
]

# ============================================================
# 5) Generate local parameter scan ranges
# ============================================================
function generate_param_ranges(baseline, base_ranges, percentage=nothing)
    n_params = length(baseline)
    ranges = Vector{Tuple{Float64, Float64}}(undef, n_params)

    for i in 1:n_params
        if isnothing(percentage)
            ranges[i] = base_ranges[i]
        else
            base_val = baseline[i]
            delta = abs(base_val * percentage)
            low = max(base_val - delta, base_ranges[i][1])
            high = min(base_val + delta, base_ranges[i][2])
            ranges[i] = (min(low, high), max(low, high))
        end
    end

    return ranges
end

# ============================================================
# 6) OAT sensitivity analysis
# ============================================================
function oat_sensitivity(model_function, baseline_params, param_ranges_to_scan; n_points=11)
    n_params = length(baseline_params)
    sensitivities = zeros(Float64, n_params)
    normalized_sensitivities = zeros(Float64, n_params)
    param_values_dict = Dict{Int, Vector{Float64}}()
    error_values_dict = Dict{Int, Vector{Float64}}()

    baseline_error = model_function(baseline_params)
    println("Baseline loss: ", baseline_error)

    for i in 1:n_params
        low, high = param_ranges_to_scan[i]
        current_param_values = collect(range(low, high, length=n_points))

        errors = Float64[]
        for val in current_param_values
            params_mod = copy(baseline_params)
            params_mod[i] = val
            push!(errors, model_function(params_mod))
        end

        param_values_dict[i] = current_param_values
        error_values_dict[i] = errors

        max_change = maximum(abs.(errors .- baseline_error))
        sensitivities[i] = max_change

        param_range_val = high - low
        normalized_sensitivities[i] = param_range_val > 1e-12 ? max_change / param_range_val : 0.0

        println("Processed parameter $i: $(param_names[i])")
    end

    return sensitivities, normalized_sensitivities, param_values_dict, error_values_dict, baseline_error
end

# ============================================================
# 7) Plot helpers
# ============================================================
function plot_top_sensitivities(sensitivities_vec, param_names_vec, range_label; n_top=8, filename_prefix="oat_demo")
    sorted_idx = sortperm(sensitivities_vec, rev=true)
    top_idx = sorted_idx[1:min(n_top, length(sorted_idx))]

    p = bar(
        param_names_vec[top_idx],
        sensitivities_vec[top_idx],
        title="Top $(length(top_idx)) OAT Sensitivities ($range_label)",
        xlabel="Parameter",
        ylabel="Normalized Sensitivity",
        legend=false,
        rotation=45,
        bottom_margin=15mm,
        xgrid=false
    )

    savefig(p, "$(filename_prefix)_top_sensitivities_$(range_label).png")
    return top_idx
end

function plot_sensitivity_curves(param_values_dict, error_values_dict, param_names_vec, top_indices, baseline_error, range_label; n_top_plots=4, filename_prefix="oat_demo")
    num_to_plot = min(n_top_plots, length(top_indices))
    layout_val = num_to_plot <= 2 ? (1, num_to_plot) : (2, ceil(Int, num_to_plot / 2))

    p = plot(layout=layout_val, size=(400 * layout_val[2], 300 * layout_val[1]), margin=5mm)

    for (plot_idx, param_idx) in enumerate(top_indices[1:num_to_plot])
        plot!(
            p[plot_idx],
            param_values_dict[param_idx],
            error_values_dict[param_idx],
            marker=:circle,
            linewidth=2,
            legend=false,
            xlabel="Parameter value",
            ylabel="Loss",
            title=param_names_vec[param_idx]
        )
        hline!(p[plot_idx], [baseline_error], linestyle=:dash)
    end

    plot!(p, plot_title="OAT Sensitivity Curves ($range_label)")
    savefig(p, "$(filename_prefix)_curves_$(range_label).png")
end

# ============================================================
# 8) Run analyses
# ============================================================
baseline_params = copy(true_params)
variation_percentages = [0.1, 0.5]

all_results = Dict()

println("\nRunning OAT sensitivity analysis for full ranges...")
sens_full, norm_sens_full, param_vals_full, error_vals_full, base_err_full =
    oat_sensitivity(model_loss, baseline_params, parameter_ranges, n_points=11)

top_idx_full = plot_top_sensitivities(norm_sens_full, param_names, "full_range")
plot_sensitivity_curves(param_vals_full, error_vals_full, param_names, top_idx_full, base_err_full, "full_range")

all_results["full_range"] = (
    sens_full, norm_sens_full, param_vals_full, error_vals_full, top_idx_full, base_err_full
)

for percentage in variation_percentages
    pct_label = @sprintf("%dpct", Int(percentage * 100))
    println("\nRunning OAT sensitivity analysis for $(pct_label)...")

    current_ranges = generate_param_ranges(baseline_params, parameter_ranges, percentage)

    sens, norm_sens, param_vals, error_vals, base_err =
        oat_sensitivity(model_loss, baseline_params, current_ranges, n_points=11)

    top_idx = plot_top_sensitivities(norm_sens, param_names, pct_label)
    plot_sensitivity_curves(param_vals, error_vals, param_names, top_idx, base_err, pct_label)

    all_results[pct_label] = (sens, norm_sens, param_vals, error_vals, top_idx, base_err)
end

println("\nOAT demo analysis complete.")

# ============================================================
# 9) Export summary table
# ============================================================
open("oat_demo_top_parameters.csv", "w") do io
    println(io, "RangeLabel,Rank,ParameterIndex,ParameterName,Sensitivity,NormalizedSensitivity,BaselineLoss")

    for (range_label, result_tuple) in all_results
        sens, norm_sens, _, _, top_idx, base_err = result_tuple
        for (rank, idx) in enumerate(top_idx)
            println(io,
                "$range_label,$rank,$idx,$(param_names[idx]),$(sens[idx]),$(norm_sens[idx]),$base_err")
        end
    end
end

println("Saved summary to oat_demo_top_parameters.csv")
