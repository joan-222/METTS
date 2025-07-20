import tn_julia: cpscollapse, cps_z, ctm, mpo_expectation, heisenbergmpo
using Plots

ensemble_size = 50
L = 8
beta = 1
Nkeep = 15

E_in = zeros(ensemble_size)
E_f = zeros(ensemble_size)

for i in 1:ensemble_size
    cps = cps_z(1, L)
    E_in[i] = mpo_expectation(heisenbergmpo(L, 1.0), copy(cps))

    metts = ctm(copy(cps), beta, Nkeep)
    cps_new = cpscollapse(metts, 3)
    E_f[i] = mpo_expectation(heisenbergmpo(L, 1.0), copy(cps_new))

    println(i)
end

histogram(E_in)
histogram(E_f)