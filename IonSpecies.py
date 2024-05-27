# Ion species class

from ADAS import data, rates
from pathlib import Path
import ICS
import h5py
import numpy as np
from scipy.constants import c


class IonSpecies:
    
    
    def __init__(self, name, Z, n):
        """
        Constructor.
        """
        self.name = name
        self.Z = Z
        self.n = n

        self.solution = n

        # Load ADAS rates
        self.acd = rates.get_rate(name, 'acd')
        self.ccd = rates.get_rate(name, 'ccd')
        self.plt = rates.get_rate(name, 'plt')
        self.prb = rates.get_rate(name, 'prb')
        self.scd = rates.get_rate(name, 'scd')

        # Load ICS parameter fits
        with h5py.File(Path(__file__).parent / 'BCGfits.h5', 'r') as f:
            self.betaStar = f[name]['betaStar'][:]
            self.C1 = f[name]['C1'][:]
            self.DI1 = f[name]['DI1'][:]


        self.Irecache = []

        _, _, _, _, self.Wioniz = data.getIonizationData(name)


    def cacheImpactIonizationRate(self, fre):
        """
        Calculate and store the fast-electron impact ionization rate.
        """
        for Z0 in range(self.Z):
            self.Irecache.append(self.evaluateImpactIonizationRate(Z0, fre=fre, usecache=False))

        self.Irecache.append(0)


    def evaluateICS(self, p, Z0):
        """
        Evaluate the ionization cross-section for relativistic electron
        impact ionization at the given momentum (normalized to mc).
        """
        return ICS.evaluate(p=p, C=self.C1[Z0], I_pot_eV=self.DI1[Z0], betaStar=self.betaStar[Z0])


    def evaluateImpactIonizationRate(self, Z0, fre, usecache=True):
        """
        Evaluate the fast electron impact ionization rate in the given charge
        state and fast electron distribution function.
        """
        if usecache and len(self.Irecache) > 0:
            return self.Irecache[Z0]

        # Evaluate ICS
        sigma = np.zeros(fre.f.shape)

        for i in range(fre.p.size):
            sigma[:,i] = self.evaluateICS(fre.p[i], Z0)

        v = c * fre.p / np.sqrt(1+fre.p**2)
        #return v*sigma
        return fre.moment(v*sigma)


    def getNeutralDensity(self):
        """
        Evaluates the neutral density of this species.
        """
        return self.solution[0]


    def getElectronDensity(self):
        """
        Evaluates the electron density corresponding to the
        most recent 'solution'.
        """
        if self.solution is None:
            raise Exception("No solution has been specified yet.")

        ne = 0
        for Z0 in range(1, self.Z+1):
            #print(f"Z0={Z0}")
            ne += self.solution[Z0] * Z0

        return ne


    def setDensity(self, n):
        """
        Set the total density of this ion species.
        """
        self.n = n


    def setSolution(self, n):
        """
        Set the ion densities from an equation solution.
        """
        if n.size != self.Z+1:
            raise Exception(f"Invalid dimensions of solution. Expected ({self.Z+1},), got {n.shape}.")

        self.solution = n

