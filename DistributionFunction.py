# Class representing a runaway electron distribution function

import numpy as np
#from DREAM.Formulas import getAvalancheDistribution
import scipy.constants 


def getAvalancheDistribution(p, xi, E, Z, nre=1, logLambda=15):
    """
    Evaluates the analytical avalanche distribution function according to
    equation (2.17) of [Embreus et al, J. Plasma Phys. 84 (2018)].

    :param p:         Momentum grid on which to evaluate the distribution.
    :param xi:        Pitch grid on which to evaluate the distribution.
    :param E:         Electric field strength (normalized to the Connor-Hastie field, Ec).
    :param Z:         Plasma total charge (= 1/ne_tot * sum_i ni * Zi^2)
    :param nre:       Runaway electron density.
    :param logLambda: Coulomb logarithm.
    """
    if p.ndim == 1:
        P, XI = np.meshgrid(p, xi)
    else:
        P, XI = p, xi

    c = scipy.constants.c
    m_e = scipy.constants.m_e

    g = np.sqrt(1+P**2)
    A = (E+1) / (Z+1) * g
    cZ = np.sqrt(5+Z)
    g0 = cZ*logLambda

    pf = m_e*c * nre * A / (2*np.pi*m_e*c*g0*P**2) / (1-np.exp(-2*A))
    f = pf * np.exp(-g/g0 - A*(1-XI))

    return f



class DistributionFunction:


    p = None
    xi = None
    f = None

    dp = None
    dxi = None


    def __init__(self):
        """
        Constructor.
        """
        pass


    def _generateGrid(self, pMin, pMax, nP, nXi):
        """
        Generate a uniform grid on which to evaluate the distribution function.
        """
        self.pMin = pMin
        self.pMax = pMax

        pp = np.linspace(pMin, pMax, nP+1)
        p = 0.5*(pp[:-1] + pp[1:])
        xip = np.linspace(-1, 1, nXi+1)
        xi = 0.5*(xip[:-1] + xip[1:])

        self.p = p
        self.xi = xi
        self.dp = pp[1:] - pp[:-1]
        self.dxi = xip[1:] - xip[:-1]

        self.P, self.XI = np.meshgrid(self.p, self.xi)
        self.DP  = np.repeat(self.dp.reshape((1,self.p.size)), self.xi.size, axis=0)
        self.DXI = np.repeat(self.dxi.reshape((self.xi.size,1)), self.p.size, axis=1)

        return p, xi


    def moment(self, weight):
        """
        Evaluate a moment of the distribution function with the given weight.
        """
        fac = 1
        #fac = 0.6905
        #fac = 0.805
        I = fac*2*np.pi * np.sum(weight * self.f*self.P**2*self.DP*self.DXI)
        return I


    def setAvalanche(self, nre, pMin, pMax, nP=100, nXi=90, E=2, Z=9):
        """
        Set this distribution function according to an analytical
        avalanche distribution.
        """
        p, xi = self._generateGrid(pMin=pMin, pMax=pMax, nP=nP, nXi=nXi)
        f = getAvalancheDistribution(p, xi, E=E, Z=Z, nre=nre)

        self.f = f


    def setStep(self, nre, pMin, pMax, nP=100, nXi=2, pUpper=20):
        """
        Set this distribution function to a step function in momentum
        space.

        :param pUpper: Maximum momentum of runaway electrons.
        """
        p, xi = self._generateGrid(pMin=pMin, pMax=pMax, nP=nP, nXi=nXi)

        idx = np.argmin(np.abs(p-pUpper))
        f = np.zeros((nXi, nP))
        f[:,0:(idx+1)] = 3*nre / (4*np.pi*(pUpper**3 - pMin**3))

        self.f = f


    def saveToHDF5(self, f, saveto='fre'):
        """
        Save to an HDF5 file using the given HDF5 file handle.
        """
        g = f.create_group(saveto)

        g['p'] = self.p
        g['xi'] = self.xi
        g['dp'] = self.dp
        g['dxi'] = self.dxi
        g['f'] = self.f


