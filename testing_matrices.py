

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

from Radiation_losses import RadiationLosses, Transport

from ITER import NRE
from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T


sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

# Full path to the HDF5 file
filename = 'cache\data_ionization'

with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]


ions = IonHandler()
ions.addIon('D', 1, 1e19)
ions.addIon('Ne', 10, 1e19)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold  = INTNe

'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)
pn=0.1

nfree, n = ionrate.equilibriumAtPressure(ions, pn, 2, 2*scipy.constants.e, fre)

ions.setSolution(n)
Di = 1
dist = 0.25 * 1.2

#print(ions['D'].solution)
#print(ions['Ne'].solution)

ne=1e19
Te=2

def Zeff(ions: IonHandler, ne):
    x=0
    for ion in ions:
        for j in range(ion.Z+1):
            x += j**2 * ion.solution[j]
    Zeff = x/ne
    return Zeff

Z = Zeff(ions, ne)

def construct_matrix(ions: IonHandler, ne, Te, Z, fre=None, V_plasma=1, V_vessel=1):
    """
    Construct the matrix for the ion rate equation.
    """
    N = ions.getNumberOfStates() + 2
    A = np.zeros((N, N))
    e = scipy.constants.e
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c
    pi = scipy.constants.pi
    m_e = scipy.constants.m_e
    Ec = getEc(Te, nfree)
    #print(nfree)
    
    def Zeff(ions: IonHandler, ne):
        x=0
        for ion in ions:
            for j in range(ion.Z+1):
                x += j**2 * ion.solution[j]
        Zeff = x/ne
        return Zeff

    Z = Zeff(ions, ne)
    sigma = evaluateBraamsConductivity(nfree, Te, Z)
    

    #iVf = lambda j : (V_plasma / V_vessel) if j==0 else 1
    #Zeff = Zeff(ions, nfree)
    off = 0
    dEc, dEc2, dEc_dne, dEc2_dne = getCriticalFieldDerivativeWithRespectToTemperature(Te, nfree)
    dsigma = derivative_sigma_T(nfree, Te, Z)
    transp_deriv = 0
    for ion in ions:
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
        dIdT = lambda j :  0 if j==ion.Z else ion.scd.deriv_Te(Z0=j, n=ne, T=Te)
        dRdT = lambda j :  0 if j==0     else ion.acd.deriv_Te(Z0=j, n=ne, T=Te)
        dIdne = lambda j : 0 if j==ion.Z else ion.scd.deriv_ne(Z0=j, n=ne, T=Te)
        dRdne = lambda j : 0 if j==0     else ion.acd.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedne = lambda j : 0 if j==ion.Z else ion.plt.deriv_ne(Z0=j, n=ne, T=Te)
        dL_freedne = lambda j : 0 if j==0 else ion.prb.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedT = lambda j : 0 if j==ion.Z else ion.plt.deriv_Te(Z0=j, n=ne, T=Te)
        dL_freedT = lambda j :0 if j==0 else ion.prb.deriv_Te(Z0=j, n=ne, T=Te)
        

        for j in range(ion.Z+1):
            if j > 0:
                A[off+j,off+j-1] = I(j-1)*ne

            A[off+j,off+j] = -(I(j) + R(j))*ne

            if j < ion.Z:
                A[off+j,off+j+1] = R(j+1)*ne
            
            if j > 0 and j < ion.Z :
                A[off+j, N-2] += ne*(dIdne(j-1)*ion.solution[j-1] - dIdne(j)*ion.solution[j] + dRdne(j+1)*ion.solution[j+1] - dRdne(j)*ion.solution[j]) + \
                I(j-1) * ion.solution[j-1] - I(j) * ion.solution[j] + R(j+1)*ion.solution[j+1] - R(j)*ion.solution[j]
                A[off+j, N-1] += dIdT(j-1)*ne*ion.solution[j-1] - dIdT(j)*ne*ion.solution[j] + dRdT(j+1)*ion.solution[j+1] - dRdT(j)*ion.solution[j]
            elif j==0 :
                A[off+j, N-2] +=  ne*(-dIdne(j)*ion.solution[j] + dRdne(j+1)*ion.solution[j+1] - dRdne(j)*ion.solution[j]) - I(j)*ion.solution[j] + R(j+1)*ion.solution[j+1] - R(j)*ion.solution[j]
                A[off+j, N-1] += - dIdT(j)*ne*ion.solution[j] + dRdT(j+1)*ne*ion.solution[j+1] - dRdT(j)*ne*ion.solution[j]
                
            elif j==ion.Z:
                A[off+j, N-2] += ne*(dIdne(j-1)*ion.solution[j-1] - dIdne(j)*ion.solution[j] - dRdne(j)*ion.solution[j]) + I(j-1)*ion.solution[j-1] - I(j) * ion.solution[j] - R(j)*ion.solution[j]
                A[off+j, N-1] += dIdT(j-1)*ne*ion.solution[j-1] - dIdT(j)*ne*ion.solution[j] - dRdne(j)*ne*ion.solution[j]
                
            A[N-2, off+j] = j
            A[N-2, N-2] = -1
            A[N-2, N-1] = 0
            
            x = 1 / (1+Z)
            sum1 = L_line(j) + L_free(j) 
            
# Add fast-electron impact ionization
            if fre is not None:
                if j < ion.Z:
                    A[off+j,off+j] -= ion.evaluateImpactIonizationRate(Z0=j, fre=fre)
                    if j > 0:
                        A[off+j,off+j-1] += ion.evaluateImpactIonizationRate(Z0=j-1, fre=fre)
            #Replacing the equation for the charged deuterium with the neutral balance condition
            if j==1:
               A[1, off+j-1] = -Te*scipy.constants.e
        A[1, N-2] = 0
        A[1, 1] = 0
        
            
        off += ion.Z+1               
        transp_deriv += (Di/(dist**2))*ion.solution[0]*scipy.constants.e 
    A[N-3, 0:2] = 0
    A[N-3, N-1] = 0
    A[N-3, N-2] = 1
    offset=0
    
    #Replace the particle balance equation for fully ionized Ne with total Ne density
    for ion in ions:
        for j in range(ion.Z+1):
            A[N-3,offset+j+1] = 1
        offset +=ion.Z
    A[N-3, 0:2] = 0
    A[N-3, N-2:] = 0
    A[1, 2:] = 0
    A[1,0:2] =1
    
    #Set the final row for the power balance equation
    sum1=sum1*nfree
    count=0
    A[1,N-1] = 0
    '''
    for ion in ions:
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
        #A[1,N-1] -= ion.solution[0]
        for j in range(ion.Z+1):
            if j != ion.Z:
                A[N-1, count+j+1] += braams_conductivity_derivative_with_respect_to_Z(nfree, Te, Z)*(-x**2 * Z * (j**2)/2)*(getEc(Te, nfree)**2) - nfree *(L_line(j)+L_free(j) + ion.IonThreshold[j]*(I(j)-R(j+1)))
        count+=ion.Z
        '''
    '''Overwriting the last row for the power balance equation'''
    
    first_term_Pow_balance = []
    second_term_Pow_balance = []
    third_term_Pow_balance = []
    fourth_term_Pow_balance = []
    
    off = 0
    '''Calculate the derivative of the power balance equation with respect to ion densities'''
    for ion in ions:
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
        

# =============================================================================
#         for j in range(ion.Z+1):
#             second_term_Pow_balance.append(braams_conductivity_derivative_with_respect_to_Z(nfree, Te, Z) * (j**2/ne)*(getEc(Te, nfree)**2))
#             third_term_Pow_balance.append(L_free(j)+L_line(j))
#             if j != ion.Z:
#                 third_term_Pow_balance[off+j] += e*ion.IonThreshold[j]*(I(j)-R(j))
#             third_term_Pow_balance[off+j] = third_term_Pow_balance[off+j]*nfree
#             if j==0:
#                 fourth_term_Pow_balance.append(Di/(dist**2)*(Te-0.025)*scipy.constants.e)
#             else:
#                 fourth_term_Pow_balance.append(0)
#                 
#             A[N-1, off+j] = second_term_Pow_balance[j] - fourth_term_Pow_balance[j] - third_term_Pow_balance[j]
#         off += ion.Z+1
# =============================================================================
        
        for j in range(ion.Z+1):
            second_term_Pow_balance.append(braams_conductivity_derivative_with_respect_to_Z(nfree, Te, Z) * (j**2/ne)*(getEc(Te, nfree)**2))
            if j != ion.Z:
                third_term_Pow_balance.append(I(j) * e * ion.IonThreshold[j] + L_line(j))
            else:
                third_term_Pow_balance.append(0)
            
            if j==0:
                fourth_term_Pow_balance.append((Di/(dist**2))*(Te-0.025)*e)
            else:
                fourth_term_Pow_balance.append(0)
            
            A[N-1, off+j] = second_term_Pow_balance[off+j] - fourth_term_Pow_balance[off+j] - third_term_Pow_balance[off+j] * nfree
        off += ion.Z+1
            
            
    
    '''Calculate the derivative of the power balance equation with respect to electron temperature'''
    third_term_deriv_Te =[]
    first_term_deriv_Te = e * c * NRE * dEc
    second_term_deriv_Te = sigma*dEc2 + dsigma*Ec**2
    fourth_term_deriv_Te = 0
    Rad_deriv_Te = 0
    off=0
    for ion in ions:
        dIdT = lambda j :  0 if j==ion.Z else ion.scd.deriv_Te(Z0=j, n=ne, T=Te)
        dRdT = lambda j :  0 if j==0     else ion.acd.deriv_Te(Z0=j, n=ne, T=Te)
        dIdne = lambda j : 0 if j==ion.Z else ion.scd.deriv_ne(Z0=j, n=ne, T=Te)
        dRdne = lambda j : 0 if j==0     else ion.acd.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedne = lambda j : 0 if j==ion.Z else ion.plt.deriv_ne(Z0=j, n=ne, T=Te)
        dL_freedne = lambda j : 0 if j==0 else ion.prb.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedT = lambda j : 0 if j==ion.Z else ion.plt.deriv_Te(Z0=j, n=ne, T=Te)
        dL_freedT = lambda j :0 if j==0 else ion.prb.deriv_Te(Z0=j, n=ne, T=Te)

        for j in range(ion.Z+1):
            if j == 0:
                fourth_term_deriv_Te += (Di/(dist**2)) * scipy.constants.e * ion.solution[0]
            if j != ion.Z:
                third_term_deriv_Te.append((dIdT(j)*e*ion.IonThreshold[j] + dL_linedT(j))*ion.solution[j] + (dL_freedT(j+1) - dRdT(j+1)*e*ion.IonThreshold[j])*ion.solution[j+1])
               #Rad_deriv_Te += third_term_deriv_Te[off+j]
            else: 
                third_term_deriv_Te.append(0)
            Rad_deriv_Te +=third_term_deriv_Te[off+j]
               
# =============================================================================
#             third_term_deriv_Te.append(dL_freedT(j)+dL_linedT(j))
#             if j != ion.Z:
#                 third_term_deriv_Te[off+j] += e*ion.IonThreshold[j]*(dIdT(j)-dRdT(j))
#             third_term_deriv_Te[off+j] = third_term_deriv_Te[off+j]*ion.solution[j]*nfree
#             Rad_deriv_Te += third_term_deriv_Te[off+j]
# =============================================================================
        off +=ion.Z+1
    A[N-1, N-1] = first_term_deriv_Te + second_term_deriv_Te - fourth_term_deriv_Te - Rad_deriv_Te*nfree      
    '''Calculate the derivative of the power balance equation with respect to electron density'''   
    first_term_deriv_ne = e**4 * c * NRE * getCoulombLogarithm(Te, nfree) / (4*pi*eps0**2 *m_e*c**2)
    
    off = 0
    dZdne = 0
    for ion in ions:
        for j in range(ion.Z+1):
            dZdne -= j**2 * ion.solution[j]
        off +=ion.Z+1
    dZdne = dZdne/(ne**2)
    
    second_term_deriv_ne =  sigma * e**6 * (2*ne) * getCoulombLogarithm(Te, nfree) **2 / (16 * pi**2 * eps0**4 *m_e**2 * c**4) + braams_conductivity_derivative_with_respect_to_Z(nfree, Te, Z)*dZdne * getEc(Te, nfree)**2
    '''radiation term to be added'''
    third_term_deriv_ne = []
    Rad_deriv_ne = 0
    off=0
    for ion in ions:
        dIdne = lambda j : 0 if j==ion.Z else ion.scd.deriv_ne(Z0=j, n=ne, T=Te)
        dRdne = lambda j : 0 if j==0     else ion.acd.deriv_ne(Z0=j, n=ne, T=Te)
        dL_linedne = lambda j : 0 if j==ion.Z else ion.plt.deriv_ne(Z0=j, n=ne, T=Te)
        dL_freedne = lambda j : 0 if j==0 else ion.prb.deriv_ne(Z0=j, n=ne, T=Te)  
        for j in range(ion.Z+1):
            if j != ion.Z:
                Rad_deriv_ne += (I(j)*e*ion.IonThreshold[j]+L_line(j))*ion.solution[j] + (L_free(j+1)-R(j+1)*e*ion.IonThreshold[j])*ion.solution[j] +\
                ne*((dIdne(j)*e*ion.IonThreshold[j] +dL_linedne(j))*ion.solution[j]+(dL_freedne(j+1)-dRdne(j+1)*e*ion.IonThreshold[j])*ion.solution[j+1])
# =============================================================================
#             third_term_deriv_ne.append(dL_freedne(j)+dL_linedne(j))
#             if j != ion.Z:
#                 third_term_deriv_ne[off+j]+=e*ion.IonThreshold[j]*(dIdne(j)-dRdne(j))
#             third_term_deriv_ne[off+j] = third_term_deriv_ne[off+j]*ion.solution[j]*nfree
#             Rad_deriv_ne += third_term_deriv_ne[off+j]
# =============================================================================
        off += ion.Z+1    
    A[N-1,N-2] = first_term_deriv_ne + second_term_deriv_ne - Rad_deriv_ne
    
    #print(f'First term deriv ne ={first_term_deriv_ne}')
    #print(f'Second term deriv ne ={second_term_deriv_ne}')
    #print(f'Fourth term deriv Te ={fourth_term_deriv_ne}')
    #print(f'Third term deriv ne ={Rad_deriv_ne}')
    
   
    return A



def construct_F(ions: IonHandler, ne, Te, Z, fre=None):
    N = ions.getNumberOfStates()+2
    F = np.zeros((N,))
    Ec = getEc(Te, nfree)
    e = scipy.constants.e
    eps0 = scipy.constants.epsilon_0
    c = scipy.constants.c
    pi = scipy.constants.pi
    m_e = scipy.constants.m_e
    Di = 1
    Dist = 0.25*1.2
    sigma = evaluateBraamsConductivity(nfree, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, Dist, Te, Tw=0.025)
    off=0
    F[N-2] = -ne
    F[N-1] = e*c*NRE*Ec +sigma*(Ec**2) - Prad - Ptransp
    for ion in ions:
        I = lambda j :     0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda j :     0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        L_line = lambda j : 0 if j==ion.Z else ion.plt(Z0=j, n=ne, T=Te)
        L_free = lambda j : 0 if j==0     else ion.prb(Z0=j, n=ne, T=Te)
        
        for j in range(ion.Z+1):
            if j == 0:
                F[off+j] = -(I(j)*ne + ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] + R(j+1)*ne*ion.solution[j+1] -R(j)*ne*ion.solution[j]
            elif j==ion.Z :
                F[off+j] = (I(j-1)*ne +ion.evaluateImpactIonizationRate(Z0=j-1,fre=fre))*ion.solution[j-1] - (I(j)*ne +ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] - R(j)*ne*ion.solution[j]
            else:
                F[off+j] = (I(j-1)*ne +ion.evaluateImpactIonizationRate(Z0=j-1,fre=fre))*ion.solution[j-1] - (I(j)*ne +ion.evaluateImpactIonizationRate(Z0=j,fre=fre))*ion.solution[j] +R(j+1)*ne*ion.solution[j+1] - R(j)*ne*ion.solution[j]
            F[N-2] += j*ion.solution[j] 
            
           
        off += ion.Z+1
    for ion in ions:
        if ion.name == 'D':
            F[1] = -ion.n
            for j in range(ion.Z+1):
                F[1] += ion.solution[j]
        if ion.name not in ['H', 'D']:
            F[N-3] = -ion.n
            x = 0 
            for j in range(ion.Z+1):
                x += ion.solution[j]
                
            #print(x-ion.n)
            F[N-3] = x-ion.n  
    
    return F

def construct_b(ions, ne, Te, np):
    N = ions.getNumberOfStates()+2
    b = np.zeros(N)
    off = 0 
    b[-1] = 0
    b[-2] = ne
    for ion in ions:
        b[off+ion.Z] = ion.n 
        off +=ion.Z+1
    return b



    

#test,b = ionrate.construct_matrix(ions, ne, Te)
A = construct_matrix(ions, ne, Te, Z)
F = construct_F(ions, ne, Te, Z)
B = construct_b(ions, ne, Te, np)
#B=B[:-2]

A = A[:-2, :-2]
test, testb = ionrate.construct_matrix(ions, ne, Te)
B = B[:-2]
x=solve(A,B)
print(x)


Te_arr = np.linspace(1,10,100)
dat1 = []
dat2 = []
dat3 = []
dat4 = []

e=scipy.constants.e
c=scipy.constants.c

for Te in Te_arr:
    
    nfree, n = ionrate.equilibriumAtPressure(ions, pn, Te, Te*e, fre)
    ions.setSolution(n)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, 0.25*1.2, Te, Tw=0.025)
    print(Ptransp)
    
    Ec = getEc(Te, ne)
    sigma = evaluateBraamsConductivity(nfree, Te, Z)
    dat1.append(e*c*NRE*Ec)
    dat2.append(Prad)
    dat3.append(Ptransp)
    dat4.append(sigma*(Ec**2))

dat1 = np.array(dat1)
dat2 = np.array(dat2)
dat3 = np.array(dat3)
dat4 = np.array(dat4)

#print(Prad)
#print(test)
plt.plot(Te_arr, dat1,label='Heating')
plt.plot(Te_arr, dat2,label='losses')
plt.plot(Te_arr, dat3,label='Neutral transport')
plt.plot(Te_arr, dat4, label = 'Ohmic heating')
plt.legend()
plt.yscale('log')

'''
plt.subplot(122)
plt.plot(Te,Ptransp)
'''

