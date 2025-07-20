function cpscollapse(metts, base)
    L = size(metts, 1)

    for i in 1:L
        # finding probabilities pm
        pm = zeros(3)
        for m = -1:1:1
            P = P_i(m, base)
            mpo = P_mpo(P, L, i)
            pm[m+2] = mpo_expectation(mpo, metts) # probability for each collapse
        end
        pm = pm / sum(pm) # normalise probabilities just in case
        sumprob = cumsum(pm)

        proll = rand() # generates random in (0,1)

        for j in 1:length(pm)
          if proll < sumprob[j] #then outcome is j-2
            m = j-2
            # metts = sitecanonical(metts,i) #sitecanonical at site i
            
            bond_right = metts[i][1,j,:]
            if i < L
                size_iplus1 = (1, size(metts[i+1])[2], size(metts[i+1])[3]) # new dimensions of the next site
                metts[i+1] = reshape(contract(bond_right, 1, metts[i+1], 1), size_iplus1) # contract bond with next site
            end

            c_vec = eigvecs(Sxyz(1,3))[:,j] #get |m>
            metts[i] = reshape(c_vec,1,3,1) #reshape to rank-3
            break
          end
        end
    end
    return metts
end

function P_i(m, base) # m are the eigvenvalues, takes values -1, 0, 1, returns |m><m|, base is 1:x, 2:y, 3:z
    spin = 1

    S = spinlocalspace(rationalize(spin))
    Sxyz = []

    Sx = (S[1] + S[2])/2 # site tensor for Sx
    Sy = (S[1] - S[2]) / (2im) # site tensor for Sy

    push!(Sxyz, Sx)
    push!(Sxyz, Sy)
    push!(Sxyz, S[3])

    # getting set of states ket(m)
    mvecs = eigvecs(Sxyz[base])

    m_ind = m + 2 # so that we can use m_ind as index of mvecs
    mvec = mvecs[:, m_ind] # the state ket(m)

    # creating projector P_i = ket(m) bra(m)
    P_i = LinearAlgebra.kron(conj(mvec)', mvec)

    # reshaping to produce local operator to be inserted in MPO
    P_i = reshape(P_i, 1, 3, 1, 3)

    return P_i
end

# creates a MPO with identity on all sites except i, which has local operator P
function P_mpo(P, L, i)
    d = size(P, 2)
    mpo = Array{Any}(undef, L)
    for n in 1:i-1
        mpo[n] = reshape(LinearAlgebra.I(d), 1, 3, 1, 3)
    end
    mpo[i] = P
    for n in i+1:L
        mpo[n] = reshape(LinearAlgebra.I(d), 1, 3, 1, 3)
    end

    return Vector{Array{ComplexF64,4}}(mpo) #force compatibility with expectation function

end

# --------------------

function mpo_on_mps(mpo, mps)
    L = size(mps, 1)
    mps = sitecanonical(mps, 1)
    for i in 1:L
        mps[i] = applyHtoC(mpo, mps, i)
        if i < L
            mps = sitecanonical(mps, i+1)
        end
    end
    
    return mps
end


function computeLeftEnvironment(MPO, MPS, ell_max)
    Lenv = ones((eltype(MPO[1])), (1, 1, 1))  # identity 3-leg tensor

    for ell in 1:ell_max
        Lenv = updateLeft(Lenv, MPS[ell], MPO[ell], MPS[ell])
    end

    return Lenv
end

function computeRightEnvironment(MPO, MPS, ell_min)
    L = length(MPO)
    Renv = ones((eltype(MPO[1])), (1, 1, 1))  # identity 3-leg tensor

    for ell in reverse(ell_min:L)
        Renv = updateLeft(Renv, permutedims(MPS[ell], (3,2,1)), permutedims(MPO[ell], (3,2,1,4)), permutedims(MPS[ell], (3,2,1)))
    end

    return Renv
end

function applyHtoC(W, MPS, ell)
    Lenv = computeLeftEnvironment(W, MPS, ell-1)
    Renv = computeRightEnvironment(W, MPS, ell+1)
    Cell = MPS[ell]
    Well = W[ell]

    HC = contract(Lenv,[3], Cell, [1])
    HC = contract(HC, [2,3], Well,[1,4])
    HC = contract(HC,[2,4], Renv,[3,2])

    return HC
end

function Sxyz(spin, index)
    S = spinlocalspace(rationalize(spin))
    Sxyz = []

    Sx = (S[1] + S[2])/ 2 # site tensor for Sx
    Sy = (S[1] - S[2]) / (2im) # site tensor for Sy

    push!(Sxyz, Sx)
    push!(Sxyz, Sy)
    push!(Sxyz, S[3])
   
    return Sxyz[index]
end