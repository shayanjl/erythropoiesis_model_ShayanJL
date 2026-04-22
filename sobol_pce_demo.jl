using QuasiMonteCarlo       # Sobol sampling
using Base.Threads          # for @threads
using SurrogatesPolyChaos   # PCE surrogate
using Surrogates            # for GaussOrthoPoly alias
using LinearAlgebra         # BLAS threading (optional)

# (Optional) parallel BLAS
BLAS.set_num_threads(Threads.nthreads())

# 1) Toy model
f(x) = 1*x[1]^2 + x[2] + 0.1*x[3] + 0.1*x[4]^2

# 2) Parameter space bounds
d = 4
lb = fill(0.0, d)
ub = fill( 1.0, d)  # Marc: the point here is you have to keep this parameter between 0 and 1, and if you wanted to make it like 10, you have to do a rescaling first in your previous codes!

# 3) Generate 200 Sobol QMC points (Shift-scrambled)
N = 200
A, B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, SobolSample(Shift()), 2)
X = hcat(A, B)          # 4×400 matrix
n = size(X, 2)

# 4) Evaluate f at each sample in parallel
Y = Vector{Float64}(undef, n)
@threads for i in 1:n
    Y[i] = f(view(X, :, i))
end

# 5) Build PCE surrogate of total degree 2
#    number of PCE terms = binomial(d+2,2) = 15
deg = 3
orthos = SurrogatesPolyChaos.MultiOrthoPoly([SurrogatesPolyChaos.GaussOrthoPoly(deg) for _ in 1:d], deg)
# SurrogatesPolyChaos accepts Vector of d-vectors:
xpoints = [Vector(X[:, i]) for i in 1:n]

pce = SurrogatesPolyChaos.PolynomialChaosSurrogate(
  xpoints, Y, lb, ub;
  orthopolys = orthos
)

# 6) Extract Sobol indices analytically
coeffs   = pce.coeff
multiidx = pce.orthopolys.ind    # 15×4 exponents
varY     = sum(coeffs[2:end].^2)

# first-order and total-order
S1 = zeros(d)
ST = zeros(d)
for k in 2:length(coeffs)
    exps = multiidx[k, :] .> 0
    vars = findall(exps)
    if length(vars)==1
        S1[vars[1]] += coeffs[k]^2
    end
    for j in vars
        ST[j] += coeffs[k]^2
    end
end
S1 ./= varY
ST ./= varY

# second-order interactions
S2 = zeros(d,d)
for k in 2:length(coeffs)
    vars = findall(multiidx[k, :] .> 0)
    if length(vars)==2
        i,j = vars
        S2[i,j] = S2[j,i] = coeffs[k]^2
    end
end
S2 ./= varY

# 7) Report
println("First-order indices:  ", round.(S1, digits=5))
println("Total-order indices: ", round.(ST, digits=5))
println("Second-order matrix:\n", round.(S2, digits=5))

using GlobalSensitivity
res2 = GlobalSensitivity.gsa(f,Sobol(order=[0,1,2]),A,B)

@show res2.S1, res2.ST
