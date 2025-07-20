import tn_julia: ctm, cpscollapse, cps_z, normalise, mpo_expectation, heisenbergmpo
using Plots

L = 50
S = 1
beta = 1
steps = 10
ensemble_size = 5

# initialise energy array for Sz Sx 
E_zx = zeros(steps)

# initialise energy array for only Sz
E_z = zeros(steps)

for j in 1:ensemble_size
    cps = cps_z(1, L)

    # iteration for Sz Sx
    c = copy(cps)
    for i in 1:steps
        metts = ctm(c, beta, 10)

        # measure energy E for metts generated here
        # m = copy(metts)
        # nm = normalise(m)
        E_zx[i] = E_zx[i] + mpo_expectation(heisenbergmpo(L, 1.0), metts)
        # E_zx[i] = E_zx[i] + i # placeholder for testing

        if isodd(i)
            c = cpscollapse(metts, 1)
        else 
            c = cpscollapse(metts, 3)
        end

        println("Sz Sx ensemble state number, step number: ", j, ", ", i)
    end


    # iteration for Sz
    cz = copy(cps)
    for i in 1:steps
        metts = ctm(cz, beta, 10)

        # measure energy E for metts generated here
        # m = copy(metts)
        # nm = normalise(m)
        E_z[i] = E_z[i] + mpo_expectation(heisenbergmpo(L, 1.0), metts)
        # E_z[i] = E_z[i] + i # placeholder for testing

        cz = cpscollapse(metts, 3)
        println("Sz ensemble state number, step number: ", j, ", ", i)
    end
end

# to get energy per state per site
E_zx = E_zx / (ensemble_size * L) 
E_z = E_z / (ensemble_size * L)

# plotting the energies (change title later)
plot(1:steps, [E_z E_zx], title="energy per site e vs Steps", label=["Z only" "Z and X"], linewidth=3)
xlabel!("Step Number")
ylabel!("Energy per Site")

# -----------------------------------

# function heisenbergmpo(L::Int, J::Float64 = 1.0)
#     # Spin-1 local space operators
#     Splus, Sminus, Sz, Id = spinlocalspace(rationalize(1))
#     Sx = (Splus + Sminus) / sqrt(2)
#     Sy = (Splus - Sminus) / (1im * sqrt(2))

#     # MPO bond dimension: 5, physical dim: 3
#     W = zeros(ComplexF64, 5, 3, 5, 3)  # [left, phys_in, right, phys_out]

#     # Bulk MPO rules
#     W[1, :, 1, :] = Id

#     W[2, :, 1, :] = Sx
#     W[3, :, 1, :] = Sy
#     W[4, :, 1, :] = Sz

#     W[5, :, 2, :] = J * Sx
#     W[5, :, 3, :] = J * Sy
#     W[5, :, 4, :] = J * Sz
#     W[5, :, 5, :] = Id

#     # Edges (correct shapes)
#     W_first = W[[5], :, :, :]   # shape (1, 3, 5, 3)
#     W_last = W[:, :, [1], :]    # shape (5, 3, 1, 3)

#     # Build MPO
#     if L != 1
#         mpo = [W_first]
#         for _ in 2:(L - 1)
#             push!(mpo, copy(W))
#         end
#         push!(mpo, W_last)
#     else
#         mpo = W[[5], :, [1], :]
#     end

#     return mpo
# end