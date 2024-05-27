# Routines for solving the ion rate equation


import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import sys
import time

from IonHandler import IonHandler


sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM')



def equilibrium(ions, Te, fre, reltol=1e-3, V_plasma=1, V_vessel=1):
    """
    Calculates the ion equilibrium for all species at the given temperature,
    with the given RE distribution function.
    """
    # Initial guess for electron density
    # (assume all species singly ionized on average)
    ne0 = ions.getTotalIonDensity()
    #neprev = ne0

    # Initial step
    A, b = construct_matrix(ions, ne0, Te, fre=fre, V_plasma=V_plasma, V_vessel=V_vessel)
    n = solve(A, b)

    ions.setSolution(n)
    ne = ions.getElectronDensity()

    # Solve equilibrium using a bisection algorithm
    itr = 0
    a, b = 0, ions.getTotalElectronDensity()

    while abs(a/b-1) > reltol:
        itr += 1

        ne = 0.5*(a+b)

        _A, _b = construct_matrix(ions, ne, Te, fre=fre, V_plasma=V_plasma, V_vessel=V_vessel)
        n = solve(_A, _b)
        ions.setSolution(n)

        nenew = ions.getElectronDensity()
        if nenew > ne:
            a = ne
        else:
            b = ne

    ne = 0.5*(a+b)
    A, b = construct_matrix(ions, ne, Te, fre=fre, V_plasma=V_plasma, V_vessel=V_vessel)
    n = solve(A, b)
    ions.setSolution(n)

    if abs(ne-ions.getElectronDensity()) > 2*reltol*ne:
        raise Exception(f"Bisection algorithm failed to converge to the correct solution. Solution ne={ne:.6e} does not agree with electron density from quasi-neutrality: {ions.getElectronDensity():.6e}.")
    
    #print(f'Iterations {itr}')
    return ions.getSolution()


def construct_matrix(ions: IonHandler, ne, Te, fre=None, V_plasma=1, V_vessel=1):
    """
    Construct the matrix for the ion rate equation.
    """
    N = ions.getNumberOfStates()
    A = np.zeros((N, N))
    b = np.zeros((N,))

    iVf = lambda j : (V_plasma / V_vessel) if j==0 else 1

    off = 0
    for ion in ions:
        I = lambda    j : 0 if j==ion.Z else ion.scd(Z0=j, n=ne, T=Te)
        R = lambda    j : 0 if j==0     else ion.acd(Z0=j, n=ne, T=Te)
        dIdT = lambda j : 0 if j==ion.Z else ion.scd.deriv_Te(Z0=j, n=ne, T=Te)
        dRdT = lambda j : 0 if j==0     else ion.acd.deriv_Te(Z0=j, n=ne, T=Te)
        dIdn = lambda j : 0 if j==ion.Z else ion.scd.deriv_ne(Z0=j, n=ne, T=Te)
        dRdn = lambda j : 0 if j==0     else ion.acd.deriv_ne(Z0=j, n=ne, T=Te)

        for j in range(ion.Z+1):
            if j > 0:
                A[off+j,off+j-1] = iVf(j)*I(j-1)*ne

            A[off+j,off+j] = -iVf(j)*(I(j) + R(j))*ne

            if j < ion.Z:
                A[off+j,off+j+1] = iVf(j)*R(j+1)*ne

            # Add fast-electron impact ionization
            if fre is not None:
                if j < ion.Z:
                    A[off+j,off+j] -= iVf(j)*ion.evaluateImpactIonizationRate(Z0=j, fre=fre)
                    if j > 0:
                        A[off+j,off+j-1] += iVf(j)*ion.evaluateImpactIonizationRate(Z0=j-1, fre=fre)

        A[off+ion.Z,off:(off+ion.Z+1)] = 1
        b[off+ion.Z] = ion.n

        off += ion.Z+1
    print(A)
    print(b)

    return A, b, N



def equilibriumAtPressure(ions, pn, Te, Tn, fre, n0=2e19, reltol=1e-3, species='D'):
    """
    Evaluate the equilibrium charge-state distribution at the specified
    neutral pressure.

    :param ions:    IonHandler object.
    :param pn:      Target neutral pressure.
    :param Te:      Electron temperature.
    :param Tn:      Neutral temperature (may be a list with one element per ion species; same order as ``ions``).
    :param fre:     Runaway electron distribution function.
    :param n0:      Initial guess for main species density.
    :param reltol:  Relative tolerance within which to determine main species density.
    :param species: Name of species to vary density of in order to change neutral pressure.
    """
    ions[species].n = n0
    n = equilibrium(ions, Te=Te, fre=fre)

    def press():
        """
        Evaluates the neutral pressure.
        """
        if np.isscalar(Tn):
            p = ions.getNeutralDensity() * Tn
        else:
            p = 0
            for ion, T in zip(ions, Tn):
                p += ion.getNeutralDensity() * T
        
        return p

    nS0 = ions[species].getNeutralDensity()
    nS  = ions[species].n

    # Construct better guess
    pk = press()
    if np.isscalar(Tn):
        Tnavg = Tn
    else:
        Tnavg = sum(Tn) / len(Tn)

    dnS = nS/nS0 * (pn-pk) / Tnavg

    if dnS < 0 and abs(dnS/nS) > 1:
        dnS = -0.9*nS

    # Bisection algorithm
    ions[species].n = nS+dnS
    n = equilibrium(ions, Te=Te, fre=fre)
    pk2 = press()

    # Set upper limit
    if pk > pn:
        b = nS
    elif pk2 > pn:
        b = nS+dnS
    else:
        # Iteratively find upper limit
        itr = 0
        nSk = nS
        while pk2 < pn and itr <= 10:
            nSk *= 2
            ions[species].n = nSk
            n = equilibrium(ions, Te=Te, fre=fre)
            pk2 = press()

            itr += 1

        if itr > 10:
            raise Exception("Unable to find an upper limit for the bisection algorithm. The desired neutral pressure cannot be reached.")

        b = nSk

    # Set lower limit
    if pk < pn and pk2 < pn:
        a = max(pk, pk2)
    elif pk < pn:
        a = nS
    elif pk2 < pn:
        a = ions[species].n
    else:
        a = 1e10

    nS = 0.5*(a+b)

    # Do bisection
    while abs(a/b-1) > reltol:
        ions[species].n = nS
        n = equilibrium(ions, Te=Te, fre=fre)

        if press() > pn:
            b = nS
        else:
            a = nS

        nS = 0.5*(a+b)

    if abs(press()/pn-1) > max(1e-2, 10*reltol):
        raise Exception("Unable to reach the desired neutral pressure.")

    ions.setSolution(n)
    nfree = ions.getElectronDensity()

    return nfree, n




