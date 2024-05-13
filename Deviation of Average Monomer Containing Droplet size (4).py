import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad
import math
from scipy.special import erf
import pandas as pd

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
s = delta  # Shape parameter for scipy's lognorm
scale = np.exp(mu)  # Scale parameter for scipy's lognorm

# Define the theoretical model function
def expected_monomer_size(a):
    """Calculate the expected size of droplets picking up exactly one dopant."""
    integral_numerator = lambda n: n * lognorm.pdf(n, s, scale=scale) * (a * n**(2/3) * np.exp(-a * n**(2/3)))
    integral_denominator = lambda n: lognorm.pdf(n, s, scale=scale) * (a * n**(2/3) * np.exp(-a * n**(2/3)))
    numerator, _ = quad(integral_numerator, 0, 10000000)
    denominator, _ = quad(integral_denominator, 0, 10000000)
    return numerator / denominator if denominator else mean

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

# Monte Carlo simulation with and without Boiloff effect
def Simulate(num_droplets, a, with_boiloff=True):
    init_sizes = np.random.lognormal(mu, delta, num_droplets)
    monomer_sizes = []

    for n in init_sizes:
        if np.random.random() < (a * n**(2/3) * np.exp(-a * n**(2/3))):
            n = Boiloff(n) if with_boiloff else n
            if n > 0:
                monomer_sizes.append(n)

    if not monomer_sizes:
        return np.mean(init_sizes), 0  # No monomer droplets, return average and 0

    return np.mean(init_sizes), np.mean(monomer_sizes)

# Run the simulation and calculate deviations
def Run_simulation_and_deviation(num_droplets, a_start, a_end, num_times, with_boiloff=True):
    a_values = np.linspace(a_start, a_end, num_times)
    deviations = []
    for a in a_values:
        avg_N, avg_monomer_N = Simulate(num_droplets, a, with_boiloff)
        deviation = avg_N - avg_monomer_N  # Calculate deviation
        deviations.append(deviation)
    return a_values, deviations

# Simulation parameters
a_start = 3e-4  # Start of a range
a_end = 2e-2  # End of a range
num_times = 50  # Number of points in the range
num_droplets = 1000  # Number of droplets for simulation

# Run simulation with and without Boiloff
a_values, deviations_with_boiloff = Run_simulation_and_deviation(num_droplets, a_start, a_end, num_times, with_boiloff=True)
a_values, deviations_without_boiloff = Run_simulation_and_deviation(num_droplets, a_start, a_end, num_times, with_boiloff=False)
theoretical_deviations = [mean - expected_monomer_size(a) for a in a_values]

# Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(a_values, deviations_with_boiloff, 'bo-', label='Monte Carlo with Boiloff')
plt.plot(a_values, deviations_without_boiloff, 'go-', label='Monte Carlo without Boiloff')
plt.plot(a_values, theoretical_deviations, 'r-', label='Theoretical Model')
plt.title('Comparison of Droplet Size Deviations')
plt.xlabel('Pickup Probability Parameter (a)')
plt.ylabel('Deviation from <N>')
plt.legend()
plt.grid(True)
plt.show()

# Store simulation results in an Excel file
simulation_data = {
    'a_values': a_values,
    'deviations_with_boiloff': deviations_with_boiloff,
    'deviations_without_boiloff': deviations_without_boiloff,
    'theoretical_deviations': theoretical_deviations
}

df = pd.DataFrame(simulation_data)
excel_file_path = 'simulation_results(Python).xlsx'
df.to_excel(excel_file_path, index=False)
print(f'Simulation results saved to {excel_file_path}')