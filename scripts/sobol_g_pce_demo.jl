using QuasiMonteCarlo
using SurrogatesPolyChaos
using Statistics
using LinearAlgebra

# ------------------------------------------------------------
# Demo script: estimating total-order Sobol indices for the
# Sobol-G function using a polynomial chaos surrogate.
# This is a simplified educational example.
# ------------------------------------------------------------

# =====================================
# 1) Sobol-G test function
# =====================================
function sobol_g_batch(X, a)
    n = size(X, 2)
    result = ones(n)
    for i in 1:length(a)
        result .*= (abs.(4 .* X[i, :] .- 2) .+ a[i]) ./ (1 .+ a[i])
    end
    return result
end

# Analytical total-order Sobol indices for Sobol-G
function sobol_g_analytical_total_indices(a)
    d = length(a)
    V_i = [1 / (3 * (1 + a_i)^2) for a_i in a]
    V = prod(1 .+ V_i) - 1

    if V < 1e-12
        return zeros(d)
    end

    ST = zeros(d)
    for i in 1:d
        others = setdiff(1:d, i)
        ST[i] = V_i[i] * prod(1 .+ V_i[others]) / V
    end
    return ST
end

# Extract total-order indices from a PCE surrogate
function compute_sobol_total_indices_from_pce(pce)
    coeffs = pce.coeff
    multiidx = pce.orthopolys.ind
    d = size(multiidx, 2)

    varY = sum(coeffs[2:end].^2)
    if varY < 1e-12
        return zeros(d)
    end

    ST = zeros(d)
    for k in 2:length(coeffs)
        active_vars = findall(multiidx[k, :] .> 0)
        for i in active_vars
            ST[i] += coeffs[k]^2
        end
    end

    return ST ./ varY
end

# =====================================
# 2) Problem setup
# =====================================
d = 4
a = [0.0, 1.0, 4.5, 9.0]

lb = zeros(d)
ub = ones(d)

analytical_ST = sobol_g_analytical_total_indices(a)

println("Analytical total-order indices:")
println(round.(analytical_ST, digits=5))

# =====================================
# 3) Generate Sobol samples
# =====================================
N = 500
sampler = SobolSample(Shift())
A, B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler, 2)
X = hcat(A, B)

Y = sobol_g_batch(X, a)

# =====================================
# 4) Build degree-2 PCE surrogate
# =====================================
deg = 2
orthos = SurrogatesPolyChaos.MultiOrthoPoly(
    [SurrogatesPolyChaos.GaussOrthoPoly(deg) for _ in 1:d],
    deg
)

xpoints = [Vector(X[:, i]) for i in 1:size(X, 2)]

pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(
    xpoints, Y, lb, ub;
    orthopolys = orthos
)

estimated_ST = compute_sobol_total_indices_from_pce(pce)

println("\nEstimated total-order indices from PCE:")
println(round.(estimated_ST, digits=5))

# =====================================
# 5) Relative error
# =====================================
relative_error = norm(estimated_ST - analytical_ST) / norm(analytical_ST)

println("\nRelative error:")
println(round(relative_error, digits=6))
