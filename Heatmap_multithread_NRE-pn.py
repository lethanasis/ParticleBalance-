import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
from multiprocessing import Pool
from scipy.constants import e
from pathlib import Path
from IonHandler import IonHandler
import ionrate
from DistributionFunction import DistributionFunction
from Rewriting_nonlinear import newton_method, bisection, power_balance
from Matrices_final import Zeff

def compute_te_mtx_element(args):
    j, i, pn_value, NRE_value, n_Ne_value, INTD, INTNe, fre_params, volume = args

    # Initialize ions and set their properties
    ions = IonHandler()
    ions.addIon('D', 1, 1e20)  # Assuming constant value for D
    ions.addIon('Ne', 10, n_Ne_value)
    ions['D'].IonThreshold = INTD
    ions['Ne'].IonThreshold = INTNe

    # Set up the distribution function
    fre = DistributionFunction()
    fre.setStep(**fre_params)
    ions.cacheImpactIonizationRate(fre)

    Di = 1
    dist = 0.25 * 1.2

    # Initial guess for Te
    Te = 0.7284 + 0.351 / np.sqrt(pn_value)
    Tn = Te * e
    ions['Ne'].n = n_Ne_value

    a = 0.5
    b = 5

    # Adjust `a` within a valid range
    while True:
        try:
            ne, n = ionrate.equilibriumAtPressure(ions, pn_value, a, a * e, fre)
            break
        except Exception as l:
            print(f'Adjusting value a due to exception: {l}')
            a += 0.1
            if a >= b:  # Prevent infinite loop if a exceeds b
                print(f"Cannot find valid `a` for pn_value={pn_value}, NRE_value={NRE_value}")
                return j, i, -1

    # Adjust `b` within a valid range
    while True:
        try:
            ne, n = ionrate.equilibriumAtPressure(ions, pn_value, b, b * e, fre)
            break
        except Exception as l:
            print(f'Adjusting value b due to exception: {l}')
            b -= 0.25
            if b <= a:  # Prevent infinite loop if b drops below a
                print(f"Cannot find valid `b` for pn_value={pn_value}, NRE_value={NRE_value}")
                return j, i, -1

    if a > b:
        print(f"Invalid range for bisection: a = {a}, b = {b}")
        return j, i, -1  # Return -1 for invalid Te

    # Attempt the bisection method to find Te
    Te = bisection(power_balance, a, b, ions, pn_value, Te, Tn, fre, Di, dist, NRE_value)

    # Check if Te is valid
    if Te is None or Te < 0:  # Check if bisection returned a valid Te
        print(f"Invalid Te found: Te={Te}")
        return j, i, -1

    Tn = Te * e
    ne, n = ionrate.equilibriumAtPressure(ions, pn_value, Te, Tn, fre)
    ions.setSolution(n)

    # Calculate Z_eff and run Newton's method
    Z = Zeff(ions, ne)
    n, ne, Te, k = newton_method(ions, pn_value, ne, Te, fre, Z, Di, dist, NRE_value)

    if Te < 0:
        return j, i, -1  # Return -1 for invalid Te

    return j, i, Te

def main():
    # Constants
    Volume = 4.632
    n_Ne_value = 7.2e18 / Volume  # Assuming a constant value for n_Ne
    pn = np.logspace(np.log10(2e-2), 0.5, 10)
    NRE = np.linspace(1.6e15, 1.6e17, 10)

    # Resolve the absolute path
    relative_path_to_formulas = Path('..', 'DREAM', 'py', 'DREAM', 'Formulas')
    absolute_path_to_formulas = (Path(__file__).parent / relative_path_to_formulas).resolve()
    sys.path.append(str(absolute_path_to_formulas))

    # Full path to HDF5 file
    filename = 'cache/data_ionization'

    with h5py.File(filename, "r") as f:
        INTD = f['H']['data'][:]
        INTNe = f['Ne']['data'][:]

    # Set up distribution function parameters
    fre_params = {
        'nre': 1.6e16,  # Initial value, will be updated in the loop
        'pMin': 1,
        'pMax': 100,
        'pUpper': 40,
        'nP': 400
    }

    # Prepare arguments for parallel execution
    args = [(j, i, pn[i], NRE[j], n_Ne_value, INTD, INTNe, fre_params, Volume) for j in range(NRE.size) for i in range(pn.size)]

    # Execute in parallel using up to 16 processes
    with Pool(processes=16) as pool:
        results = pool.map(compute_te_mtx_element, args)

    # Collect results into the Te_mtx array
    Te_mtx = np.zeros((NRE.size, pn.size))
    for j, i, Te in results:
        if Te != -1:
            Te_mtx[j, i] = Te

    # Plot the results
    X, Y = np.meshgrid(pn, NRE)
    heatmap = plt.pcolormesh(X, Y, Te_mtx, cmap='hot')

    plt.colorbar(heatmap, label='Te [eV]')
    plt.xlabel('Neutral Pressure [Pa]')
    plt.ylabel('NRE')
    plt.xscale('log')
    plt.savefig("Heatmap.eps")
    plt.show()

if __name__ == '__main__':
    main()
