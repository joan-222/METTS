ensemble_size = 100
L = 10
beta = 1
Nkeep = 10
T = 1/beta

ensemble = Vector{Array}(undef, ensemble_size)

for i in 1:ensemble_size
    cps = cps_z(1, L)
    for j in 1:10
        metts = ctm(copy(cps), beta, Nkeep)
        if iseven(j)
            cps = cpscollapse(copy(metts), 3)
        else
            cps = cpscollapse(copy(metts), 1)
        end
    end
    ensemble[i] = metts
end

# calculating susceptibility
Sz = Vector{Array}(undef, ensemble_size)
Sz2 = Vector{Array}(undef, ensemble_size)

for i in 1:ensemble_size
    Sz[i] = sz(metts)

    Sz2[i] = sz2(metts)
end


susceptibility =  (mean(Sz2) - (mean(Sz))^2) / (L * T)


# --------------------------------------

function sz(mps)
    return mps
end