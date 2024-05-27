# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:15:55 2024

@author: thana
"""

import numpy as np
from numpy.linalg import solve
import os
import sys 
import scipy.constants
from IonHandler import IonHandler
import ionrate
from DistributionFunction import DistributionFunction
import h5py
import matplotlib.pyplot as plt
from scipy.linalg import solve
from Radiation_losses import RadiationLosses, Transport 
from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T
#from rewriting_matrices import construct_Jacobian, construct_F, Zeff
from Matrices_final import Zeff, construct_F, construct_Jacobian

def power_balance(Te, ne, ions : IonHandler, Di, dist, NRE):
    Z = Zeff(ions, ne)
    Ec = getEc(Te, ne)
    e = scipy.constants.e
    c = scipy.constants.c
    sigma = evaluateBraamsConductivity(ne, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, dist, Te, Tw=0.025)
    
    re_heat = e*c*NRE*Ec
    ohm_heat = sigma*Ec**2
    rad_loss = Prad
    neut_transp = Ptransp
    
    F = e*c*NRE*Ec + sigma * Ec**2 - Prad - Ptransp
    return F 

def bisection(power_balance, a, b, ions, pn, Te, Tn, fre, Di, dist, NRE, tol = 1e-3, max_iter = 100 ): #'''np.sqrt(np.finfo(np.float64).eps)'''
    c = (a+b)/2
    ne, n = ionrate.equilibriumAtPressure(ions, pn, a, a*scipy.constants.e, fre)
    ions.setSolution(n)
    
    fa = power_balance(a, ne, ions, Di, dist, NRE)
    
    ne, n = ionrate.equilibriumAtPressure(ions, pn, b, b*scipy.constants.e, fre)
    ions.setSolution(n)
    fb = power_balance(b, ne, ions, Di, dist, NRE)
    if fa*fb>0:
        #print("Error : The function hsa the same signs at the endpoints")
        #return None
        raise Exception("The function has the same sign at endpoints")
    iter_count = 0
    while iter_count < max_iter:
        c = (a+b)/2
        ne, n = ionrate.equilibriumAtPressure(ions, pn, c, c*scipy.constants.e, fre)
        ions.setSolution(n)
        #print(c)
        fc = power_balance(c, ne, ions, Di, dist, NRE)
        #print(f'a ={a}, b = {b}, c = {c}')
        #print(f' fa = {fa},  fb = {fb}, fc = {fc}')
        if (b-a)<(tol+tol*c):
            return c #root within tolerance
        elif fa*fc<0:
            b=c # Root in the left half
            fb=fc
        else:
            a=c # Root in the right half
            fa=fc 
        iter_count+=1 
        
    print('Max iterations reached')
    return None

def objective_function(x,ions,Z, Di, dist, NRE, fre):
    """Scalar objective function representing squared norm of vector-valued function F."""
    n = x[:-2]
    ions.setSolution(n)
    ne = x[-2]
    Te = x[-1]
    F = construct_F(ions, ne, Te, Z, Di, dist, NRE, fre)
    f = 0.5*np.linalg.norm(F)**2
    return f

def compute_gradient(x,ions,Z, Di, dist, NRE, fre):
    n = x[:-2]
    ions.setSolution(n)
    ne = x[-2]
    Te = x[-1]
    J = construct_Jacobian(ions, ne, Te, Z, Di, dist, NRE, fre)
    F = construct_F(ions, ne, Te, Z, Di, dist, NRE, fre)
    return J.T@F 
    

# =============================================================================
# def line_search(objective_function, compute_gradient, x, dx,ions, Z):
#     alpha = 1.0
#     c = 0.5
#     rho = 0.5
#     while objective_function(x + alpha * dx,ions,Z) > objective_function(x,ions,Z) + c * alpha * compute_gradient(x,ions,Z).T@dx and alpha > 1e-3:
#         alpha *= rho
#     return alpha
# =============================================================================



def line_search(objective_function, compute_gradient, x, dx, ions, Z, Di, dist, NRE, fre, alpha = 1, beta = 0.5, max_iter = 100):
    '''
    Perform a line search to find a suitable step size alpha using backtracking line search 
    Parameters:
        objective_function (callable) : Objective function f(x) to minimize.
        compute_gradient (callable) : Gradient function ∇f(x) of the objective function.
        x (array) : Current point where the search starts.
        dx (array) : Search direction.
        ions (IonHandler) : Ion Handler object.
        alpha : Initial step size.
        beta : Factor to decrease the step size (0<b<1).
        max_iter : Maximum number of iterations for line search.
    '''
    f0 = objective_function(x, ions, Z, Di, dist, NRE, fre)
    grad = compute_gradient(x, ions, Z, Di, dist, NRE, fre)
    
    for _ in range(max_iter):
        x_new = x + alpha*dx
        f_new = objective_function(x_new, ions, Z, Di, dist, NRE, fre)
        if f_new <= f0 + beta*alpha*np.dot(grad, dx):
            return alpha
        alpha *= 0.5
    return alpha


def newton_method(ions: IonHandler, pn, ne, Te, fre, Z, Di, dist, NRE, tol =1e-6, max_iter=1000): # np.sqrt(np.finfo(np.float64).eps)
    nfree, n = ionrate.equilibriumAtPressure(ions, pn, Te, Te*scipy.constants.e, fre)
    ions.setSolution(n)
    ne_arr = []
    Te_arr = []
    n_arr = []
    for i in range(max_iter):
        J = construct_Jacobian(ions, ne, Te, Z, Di, dist, NRE, fre)
        F = construct_F(ions, ne, Te, Z, Di, dist, NRE, fre)
        dx = np.linalg.solve(J, -F)
        
        
        alpha = line_search(objective_function, compute_gradient, np.hstack((n, ne, Te)), dx,ions, Z, Di, dist, NRE, fre)
        #alpha = 0.1
        ne += dx[-2] * alpha
        Te += dx[-1] * alpha
        n += dx[:-2] * alpha
        
        
        
        #ne_arr.append(ne)
        Te_arr.append(dx[1])
        ne_arr.append(dx[13])
        n_arr.append(F[-1])
        # Update ion densities
        ions.setSolution(n)
        #Z = Zeff(ions, ne) # Doesn't change solution much
        #print(i)
        #print(f'Progress : {100*(i+1)/max_iter}%')
        #print(F)
        #print(f' Te = {Te}')
        #print(f'Te = {Te}, alpha = {alpha}')
        #print(f' d_ne = {dx[-2]}, alpha = {alpha}, tol = {tol+tol*ne}, iter = {i}')
        if np.linalg.norm(dx[:-1]) < (tol + tol * ne) and np.linalg.norm(dx[-1]) < (tol + tol * Te): #np.all((dx[:-1])
            break
        else:
            x = np.where((dx) > (tol + tol * ne))
            y = np.where((dx[-1])>(tol+tol*Te))
            #print(x,y)
        if np.any(np.isnan(dx)):
            print('Nan values - stopping iterations')
            break
    
    return n, ne, Te, i