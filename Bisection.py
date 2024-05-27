import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import sys
import time
import scipy.constants
from IonHandler import IonHandler
import ionrate 
from DistributionFunction import DistributionFunction
import h5py

from scipy.linalg import solve

from Radiation_losses import RadiationLosses, Transport


from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T

from rewriting_matrices import construct_Jacobian, construct_F, Zeff


"""Function that defines the power balance equation to solve in order to get an initial guess for the temperature"""
def power_balance(Te, ne, ions : IonHandler, Di, dist,doprint = False):
# =============================================================================
#     if doprint==True:
#         for ion in ions:
#             print(ion.solution)
# =============================================================================
    Z = Zeff(ions, ne)
    
    Ec = getEc(Te, ne)
    e = scipy.constants.e 
    c = scipy.constants.c 
    sigma = evaluateBraamsConductivity(ne, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, dist, Te, Tw=0.025)  
    
    F = e*c*NRE*Ec + sigma * Ec**2 - Prad - Ptransp
# =============================================================================
#     if doprint == True:
#         print(ne)
# =============================================================================
# =============================================================================
#     if doprint==True:
#         for ion in ions:
#             print(f'bisection script : {ion.solution}')
# =============================================================================
    return F

def initial_guess(power_balance, a, b, tol=1e-6, max_iter=100, **kwargs):
    c = (a+b)/2
    nfree, n = ionrate.equilibriumAtPressure(ions, pn, c, c*scipy.constants.e, fre)
    ions.setSolution(n)
    ne=nfree
    fa = power_balance(a, ne,**kwargs,doprint=False)
    fb = power_balance(b, ne,**kwargs,doprint=True)
    if fa*fb>0:
        print("Error : The function has the same signs at the endpoints")
        return None
    
    iter_count = 0 
    while iter_count < max_iter:
        c = (a+b)/2 #Midpoint of interval
        fc = power_balance(c, ne,**kwargs)
        
        if abs(fc)<tol*c:
            return c #Root within tolerance
        elif fa*fc<0:
            b=c # Root is in the left half
            fb=fc
        else:
            a=c #Root in the right half
            fa=fc
        iter_count += 1
        #print(c)
        nfree, n = ionrate.equilibriumAtPressure(ions, pn, c, c*scipy.constants.e, fre)
        ions.setSolution(n)
        ne=nfree
    print('Max iterations reached')
    return None


ne = 5e19
a = 1
b = 10
sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

#Full parth to HDF5 file
filename = 'cache\data_ionization'

with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]
'''Set up Ions'''
ions = IonHandler()
ions.addIon('D', 1, 1e19)
ions.addIon('Ne', 10, 1e19)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold = INTNe

NRE = 8e15
'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)
pn=0.1

nfree, n = ionrate.equilibriumAtPressure(ions, pn, 2, 2*scipy.constants.e, fre)
ions.setSolution(n)
#print(f'in bisection script {n}')
# =============================================================================
# for ion in ions:
#     print(f'check {ion.solution}')
# =============================================================================
Di = 1
dist = 0.25 * 1.2

root = initial_guess(power_balance, a, b, Di=Di, dist=dist, ions = ions)
#print(root)
Z = Zeff(ions, ne)

def objective_function(x):
    """Scalar objective function representing squared norm of vector-valued function F."""
    n = x[:-2]
    ions.setSolution(n)
    ne = x[-2]
    Te = x[-1]
    F = construct_F(ions, ne, Te, Z)
    return 0.5 * np.linalg.norm(F)**2


def compute_gradient(x):
    """Gradient of the scalar objective function."""
    n = x[:-2]
    ions.setSolution(n)
    ne = x[-2]
    Te = x[-1]
    J = construct_Jacobian(ions, ne, Te, Z)
    F = construct_F(ions, ne, Te, Z)
    return J.T @ F


def line_search(objective_function, compute_gradient, x, dx):
    alpha = 1.0
    c = 0.5  # parameter for backtracking (0 < c < 1)
    rho = 0.5  # shrinkage factor
   
    while objective_function(x + alpha * dx) > objective_function(x) + c * alpha * compute_gradient(x).T@dx:
        alpha *= rho
    return alpha


def newton_method(ions: IonHandler, ne, Te, tol = np.sqrt(np.finfo(np.float64).eps), max_iter=10000):
    nfree, n = ionrate.equilibriumAtPressure(ions, 0.1, Te, Te*scipy.constants.e, fre)
    
    for i in range(max_iter):
        J = construct_Jacobian(ions, ne, Te, Z)
        F = construct_F(ions, ne, Te, Z)
        dx = np.linalg.solve(J, -F)
        
        
        alpha = line_search(objective_function, compute_gradient, np.hstack((n, ne, Te)), dx)
        ne += alpha * dx[-2]
        Te += alpha * dx[-1]
        n += alpha * dx[:-2]
        
        # Update ion densities
        ions.setSolution(n)
        
        if np.linalg.norm(dx[:-1]) <tol + tol * ne and np.linalg.norm(dx[-1]) <tol + tol * Te:
            break
    
    return n, ne, Te, i


Te = root
n, ne, Te, i = newton_method(ions, ne, Te)

# =============================================================================
# print(f'Densities are {n}')
# print(f'Electron density is {ne}')
# print(f'Electron temperature is {Te}')
# print(f'Number of iterations: {i}')
# =============================================================================
# =============================================================================
# print(f'number of iter = {i+1}')
# print(f'Te = {Te}')
# =============================================================================
