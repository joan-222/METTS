import tn_julia: P_mpo
import Statistics: mean

ensemble_size = 10
L = 10
beta = 10
Nkeep = 10
T = 1/beta

ensemble = Vector{Array}(undef, ensemble_size)

for i in 1:ensemble_size
    cps = cps_z(1, L)
    metts = cps
    for j in 1:10
        metts = ctm(cps, beta, Nkeep)
        if iseven(j)
            cps = cpscollapse(metts, 3)
        else
            cps = cpscollapse(metts, 1)
        end
    end
    ensemble[i] = metts
    println(i)
end

# calculating susceptibility
Sz_array = zeros(ensemble_size)
Sz2_array = zeros(ensemble_size)

for i in 1:ensemble_size
    Sz_array[i] = sz(ensemble[i])
    Sz2_array[i] = sz2(ensemble[i])
end

susceptibility =  (mean(Sz2_array) - (mean(Sz_array))^2) / (L * T)
println(susceptibility)


# --------------------------------------

function sz(mps)
    L = size(mps, 1)

    S = spinlocalspace(rationalize(1))
    Sz = S[3]
    Sz_operator = reshape(Sz, 1, 3, 1, 3)

    sztot = 0
    for i in 1:L
        P = P_mpo(Sz_operator, L, i)
        sztot = sztot + mpo_expectation(P, mps)
    end
    return sztot
end

function sz2(mps)
    L = size(mps, 1)

    S = spinlocalspace(rationalize(1))
    Sz = S[3]
    Sz2 = Sz^2
    Sz_operator = reshape(Sz2, 1, 3, 1, 3)

    sztot2 = 0
    for i in 1:L
        P = P_mpo(Sz_operator, L, i)
        sztot2 = sztot2 + mpo_expectation(P, mps)
    end
    return sztot2
end