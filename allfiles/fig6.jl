import tn_julia: ctm, cpscollapse, cps_z, normalise, mpo_expectation, heisenbergmpo
using Plots

L = 50
S = 1
beta = 1
steps = 10
ensemble_size = 20
Nkeep = 10

# initialise energy array for Sz Sx 
E_zx = zeros(steps)

# initialise energy array for only Sz
E_z = zeros(steps)

for j in 1:ensemble_size
    cps = cps_z(1, L)
    c = copy(cps)
    cz = copy(cps)

    # iteration for Sz Sx
    for i in 1:steps
        metts = ctm(c, beta, Nkeep)

        # measure energy E for metts generated here
        E = mpo_expectation(heisenbergmpo(L, 1.0), copy(metts))
        E_zx[i] = E_zx[i] + E
        println("E: ", E)

        if isodd(i)
            c = cpscollapse(metts, 1)
        else 
            c = cpscollapse(metts, 3)
        end

        println("Sz Sx ensemble state number, step number: ", j, ", ", i)
    end


    # iteration for Sz
    for i in 1:steps
        metts = ctm(cz, beta, Nkeep)

        # measure energy E for metts generated here
        Ez = mpo_expectation(heisenbergmpo(L, 1.0), copy(metts))
        E_z[i] = E_z[i] + Ez
        println("E: ", Ez)

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