using DifferentialEquations
using Plots

# ------------------------------------------------------------
# Demo script: a simplified multi-compartment cell population
# model with signaling and competition.
# This is an educational example inspired by compartmental
# biological modeling workflows, not a reproduction of any
# specific research model.
# ------------------------------------------------------------

function cell_system_demo!(du, u, p, t)
    # Parameters
    growth_stem,
    growth_prog,
    growth_precursor,
    growth_mature,
    growth_support,
    death_stem,
    death_prog,
    death_precursor,
    death_mature,
    death_support,
    diff_stem_prog,
    diff_prog_precursor,
    diff_precursor_mature,
    support_effect_prog,
    support_effect_precursor,
    disease_growth,
    disease_death,
    signal_prod,
    signal_deg,
    signal_effect,
    K = p

    # State variables
    Stem, Prog, Precursor, Mature, Support, Disease, Signal = u

    # Total population contributing to competition
    total_cells = Stem + Prog + Precursor + Mature + Support + Disease
    competition = max(0.0, 1.0 - total_cells / K)

    # Compartment dynamics
    du[1] = growth_stem * Stem * competition -
            death_stem * Stem -
            diff_stem_prog * Stem

    du[2] = diff_stem_prog * Stem +
            (growth_prog + support_effect_prog * Support) * Prog * competition -
            death_prog * Prog -
            diff_prog_precursor * Prog / (1 + signal_effect * Signal)

    du[3] = diff_prog_precursor * Prog / (1 + signal_effect * Signal) +
            (growth_precursor + support_effect_precursor * Support) * Precursor * competition -
            death_precursor * Precursor -
            diff_precursor_mature * Precursor

    du[4] = diff_precursor_mature * Precursor +
            growth_mature * Mature * competition -
            death_mature * Mature

    du[5] = growth_support * Support * competition -
            death_support * Support

    du[6] = disease_growth * Disease * competition -
            disease_death * Disease

    du[7] = signal_prod * Disease - signal_deg * Signal

    return du
end

# Initial conditions
u0 = [
    0.10,  # Stem
    0.08,  # Prog
    0.06,  # Precursor
    0.12,  # Mature
    0.07,  # Support
    0.02,  # Disease
    0.00   # Signal
]

# Synthetic parameter set for demo purposes
p = [
    0.30,  # growth_stem
    0.35,  # growth_prog
    0.28,  # growth_precursor
    0.12,  # growth_mature
    0.10,  # growth_support
    0.03,  # death_stem
    0.04,  # death_prog
    0.05,  # death_precursor
    0.03,  # death_mature
    0.02,  # death_support
    0.10,  # diff_stem_prog
    0.12,  # diff_prog_precursor
    0.15,  # diff_precursor_mature
    0.25,  # support_effect_prog
    0.20,  # support_effect_precursor
    0.40,  # disease_growth
    0.06,  # disease_death
    0.20,  # signal_prod
    0.15,  # signal_deg
    0.80,  # signal_effect
    1.00   # carrying capacity K
]

tspan = (0.0, 50.0)

prob = ODEProblem(cell_system_demo!, u0, tspan, p)
sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)

titles = [
    "Stem Cells",
    "Progenitor Cells",
    "Precursor Cells",
    "Mature Cells",
    "Support Cells",
    "Competing Population",
    "Signal Molecule"
]

plots = []

for i in 1:length(titles)
    p_i = plot(
        sol,
        vars=(0, i),
        title=titles[i],
        xlabel="Time",
        ylabel="Level",
        legend=false,
        linewidth=2
    )
    push!(plots, p_i)
end

combined_plot = plot(plots..., layout=(4, 2), size=(900, 1000))
display(combined_plot)

savefig(combined_plot, "multicompartment_model_demo.png")
