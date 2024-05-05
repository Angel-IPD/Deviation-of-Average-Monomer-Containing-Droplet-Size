import numpy as np
import math
from scipy.special import erf

# Constants and variables initialization
mean = 10**4  # mean droplet size
stdev = 9 * 10**3  # standard deviation in droplet size
pc_temp = 273.15  # pickup chamber temperature
mDop = 2.1040499999999997E-25  # dopant mass
vz = 375  # droplet velocity
mass_Helium = 6.6464764e-27  # Mass of helium atom in kg
energy_to_boil_1_He = 9.94066934e-23  # energy in J to boil off 1 He
kB = 1.38064852e-23  # Boltzmann constant

# Log-normal distribution parameters
ratio = stdev / mean
mu = math.log(mean / (math.sqrt(1 + ratio**2)))
delta = math.sqrt(math.log(1 + ratio**2))

# Boiloff function
def Boiloff(N):
    vProb = math.sqrt(2.0 * kB * pc_temp / mDop)
    xJL = vz / vProb
    mReduced = (mDop * N * mass_Helium) / (mDop + N * mass_Helium)
    eDop = kB * pc_temp * (mReduced / mDop) * (
        xJL * (5.0 / 2.0 + xJL * xJL) * math.exp(-xJL * xJL) + math.sqrt(np.pi) * 
        (3.0 / 4.0 + 3.0 * xJL * xJL + xJL ** 4.0) * erf(xJL)) / (
        xJL * math.exp(-xJL * xJL) + math.sqrt(np.pi) * (0.5 + xJL * xJL) * erf(xJL))
    dN = int(math.floor(eDop / energy_to_boil_1_He))
    return N - dN if dN < N else 0

# Probability of monomer pickup given size n
def Pickup_prob(x, a):
    return a * np.power(x, 2 / 3) * np.exp(-a * np.power(x, 2 / 3))

# Simulate function with boil-off effect
def Simulate(num_droplets, a):
    init_sizes = np.random.lognormal(mu, delta, num_droplets)
    monomer_sizes = []

    for n in init_sizes:
        if np.random.random() < Pickup_prob(n, a):
            monomer_sizes.append(Boiloff(n))

    if not monomer_sizes:
        return np.mean(init_sizes), 0  # No monomer droplets, return average and 0

    return np.mean(init_sizes), np.mean(monomer_sizes)

# Run the simulation over a range of a values and calculate deviation from <N>
def Run_simulation_and_deviation(num_droplets, a_start, a_end, num_times):
    a_values = np.linspace(a_start, a_end, num_times)
    deviations = []
    for a in a_values:
        avg_N, avg_monomer_N = Simulate(num_droplets, a)
        deviation = avg_N - avg_monomer_N  # Calculate deviation
        deviations.append(deviation)
    return a_values, deviations

# Simulation parameters
a_start = 3e-4  # Start of a range
a_end = 2e-2  # End of a range
num_times = 50  # Number of points in the range

# Run simulation and calculate deviations
a_values, deviations = Run_simulation_and_deviation(1000, a_start, a_end, num_times)

# Plotting the results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.plot(a_values, deviations, marker='o', linestyle='-')
plt.title('Deviation of Average Monomer-Containing Droplet Size' )
plt.xlabel('Pickup Probability Parameter (a)')
plt.ylabel('Deviation from <N>')
plt.grid(True)
plt.show()
