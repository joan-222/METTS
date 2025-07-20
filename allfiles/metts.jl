"""
    phi_i = ctm(psi_i, beta, Ham)

Creates a METTS state using imaginary-time evolution by 'beta/2' of a classical product state 'psi_i' using tDMRG.

# Input
- 'psi_i::AbstractVector': Classical product state. 
- 'beta::Float64': Time interval for imaginary time evolution.
- 'Nkeep::Int': Maximum allowed bond dimension after truncation.

# Output
- 'metts': Time evolved METTS.
"""

function ctm(psi_i, beta, Nkeep)
    L = size(psi_i, 1)

    # time evolving the CPS by time beta/2
    mnew = tdmrg(psi_i, beta/2, 2000, Nkeep)

    # calculating P(i)
    Id = P_mpo(reshape(LinearAlgebra.I(3), 1, 3, 1, 3), L, 1)

    P = mpo_expectation(Id, mnew)
    P = P[1]

    # calculating metts = P_i^(-1/2) * e^(- beta H / 2) psi_i
    metts = (P)^(-1/2) * mnew
    return metts
end

function tdmrg(mps, beta, Nsteps, Nkeep)
    lmps = size(mps, 1)

    # using time beta/2 for sites 1 to L-2 to get tau/2
    expHo, expHe = trotter(1, beta/2, Nsteps)
    expHoL, expHeL = trotter(1, beta, Nsteps)

    for n in 1:Nsteps
        # evolving sites using bond by bond compression
        for i in 1:(lmps-2)
            if isodd(i)
                mps = applygate(mps, i, expHo, Nkeep)
            else
                mps = applygate(mps, i, expHe, Nkeep)
            end
        end

        # for site L

        if isodd(lmps-1)
            mps = applygate(mps, (lmps-1), expHoL, Nkeep)
        else
            mps = applygate(mps, (lmps-1), expHeL, Nkeep)
        end

        # sweeping backwards
        for i in (lmps-2):-1:1
            if isodd(i)
                mps = applygate_backward(mps, i, expHo, Nkeep)
            else
                mps = applygate_backward(mps, i, expHe, Nkeep)
            end
        end
        mps = normalise(mps)

        # println("TDMRG step $n / $Nsteps")
    end

    return mps
end


function trotter_new(spin, beta, Nsteps)
    tauT = beta / Nsteps # Trotter time step

    Ho = heisenbergmpo(2)

    Ho = contract(Ho[1], 3, Ho[2], 1)
    Ho = permutedims(Ho, (1, 2, 4, 5, 6, 3)) # check perms pls
    He = Ho # even bond

    dims_Ho = size(Ho)
    dims_He = size(He)


    Homat = reshape(Ho, dims_Ho[2]*dims_Ho[3], dims_Ho[5]*dims_Ho[6])
    Hemat = reshape(He, dims_He[2]*dims_He[3], dims_He[5]*dims_He[6])


    # Matrix exponentiation
    expHo_mat = exp(-tauT/2 * Homat)
    expHe_mat = exp(-tauT/2 * Hemat)

    # Reshape back to rank-4 tensors: (left1, left2, right1, right2)
    expHo = reshape(expHo_mat, dims_Ho[2], dims_Ho[3], dims_Ho[5], dims_Ho[6])
    expHe = reshape(expHe_mat, dims_He[2], dims_He[3], dims_He[5], dims_He[6])

    return expHo, expHe
end

function trotter(spin, beta, Nsteps)
    tauT = beta / Nsteps # Trotter time step

    # Preparing the time evolution hamiltonian
    # Spin-1 local space tensor (rank-3: left-phys-right)
    S = spinlocalspace(rationalize(spin))
    Sx = (S[1] + S[2])/2 # site tensor for Sx
    Sy = (S[1] - S[2]) / (2im) # site tensor for Sy
    Sz = S[3] # site tensor for Sz
    size_s = size(Sx)[1]

    # Construct two-site Hamiltonians via tensor contraction: <S S>
    Ho = LinearAlgebra.kron(Sx, Sx) + LinearAlgebra.kron(Sy, Sy) + LinearAlgebra.kron(Sz, Sz)  # odd bond
    He = Ho # even bond

    # Matrix exponentiation
    expHo_mat = exp(-tauT/2 * Ho)
    expHe_mat = exp(-tauT/2 * He)

    # Reshape back to rank-4 tensors: (left1, left2, right1, right2)
    expHo = reshape(expHo_mat, size_s, size_s, size_s, size_s)
    expHe = reshape(expHe_mat, size_s, size_s, size_s, size_s)

    return expHo, expHe
end

function applygate(mps, l, Hgate, Nkeep)
    # contract the mps sites to produce a rank four tensor
    m = contract(mps[l], 3, mps[l+1], 1)
    # contracting the physical legs with the trotter gate
    mtau = contract(m, [2, 3], Hgate, [3, 4])
    mtau = permutedims(mtau,(1,3,4,2))

    # svd to separate back into two sites
    U, S, Vd = svd(mtau,[1, 2], Nkeep = Nkeep)

    # since the site index should move, contract S with Vd
    mps[l] = U
    mps[l+1] = contract(diagm(S), 2, Vd, 1)

    return mps
end

function applygate_backward(mps, l, Hgate, Nkeep)
    # contract the mps sites to produce a rank four tensor
    m = contract(mps[l], 3, mps[l+1], 1)

    # contracting the physical legs with the trotter gate
    mtau = contract(m, [2, 3], Hgate, [3, 4])
    mtau = permutedims(mtau,(1,3,4,2))

    # svd to separate back into two sites
    U, S, Vd = svd(mtau,[1, 2], Nkeep = Nkeep)

    # since the site index should move, contract S with Vd
    mps[l] = contract(U, 3, diagm(S), 1)
    mps[l+1] = Vd

    return mps
end

function cps_z(spin, N) #Sz product states
    localspace = spinlocalspace(rationalize(spin))
    Sz = localspace[3]  # Sz operator
    evecs = eigvecs(Sz)  # Eigenvectors in columns

    d = size(Sz, 1)  # Physical dimension (should be 2 for spin-1/2)

    mps = Vector{Array{ComplexF64,3}}(undef, N)

    for i in 1:N
        j = rand(1:d)  # Random eigenvector
        v = evecs[:, j]  # d-dimensional vector

        # Convert to MPS tensor: (left_bond=1, physical_dim=d, right_bond=1)
        mps[i] = reshape(v, 1, d, 1)
    end

    return mps
end

function heisenbergmpo(L::Int, J::Float64 = 1.0)
    # Spin-1 local space operators
    Splus, Sminus, Sz, Id = spinlocalspace(rationalize(1))
    Sx = (Splus + Sminus) / sqrt(2)
    Sy = (Splus - Sminus) / (1im * sqrt(2))

    # MPO bond dimension: 5, physical dim: 3
    W = zeros(ComplexF64, 5, 3, 5, 3)  # [left, phys_in, right, phys_out]

    # Bulk MPO rules
    W[1, :, 1, :] = Id

    W[2, :, 1, :] = Sx
    W[3, :, 1, :] = Sy
    W[4, :, 1, :] = Sz

    W[5, :, 2, :] = J * Sx
    W[5, :, 3, :] = J * Sy
    W[5, :, 4, :] = J * Sz
    W[5, :, 5, :] = Id

    # Edges (correct shapes)
    W_first = W[[5], :, :, :]   # shape (1, 3, 5, 3)
    W_last = W[:, :, [1], :]    # shape (5, 3, 1, 3)

    # Build MPO
    if L != 1
        mpo = [W_first]
        for _ in 2:(L - 1)
            push!(mpo, copy(W))
        end
        push!(mpo, W_last)
    else
        mpo = W[[5], :, [1], :]
    end

    return mpo
end

function mpo_expectation(W::Vector{<:AbstractArray{<:Number,4}}, 
                         MPS::Vector{<:AbstractArray{<:Number,3}})
    # Initialize environment as scalar identity in a 3-leg tensor form
    C = ones(eltype(W[1]), (1, 1, 1))

    for i in eachindex(W)
        C = updateLeft(C, MPS[i], W[i], MPS[i])
    end

    # At the end, C is (1,1,1), so extract scalar value
    return real(C[1,1,1])
end

function normalise(mps)
    L = length(mps)
    Id = P_mpo(reshape(LinearAlgebra.I(3), 1, 3, 1, 3), L, 1)
    norm = mpo_expectation(Id, mps)
    mps[1] = mps[1] / sqrt(norm)
    return mps
end