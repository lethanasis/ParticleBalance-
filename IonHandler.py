# Ion Handler class


import numpy as np
from IonSpecies import IonSpecies


class IonHandler:
    

    def __init__(self):
        """
        Constructor.
        """
        self.ions = {}


    def __len__(self):
        """
        Get number of ions in handler.
        """
        return len(self.ions)


    def __getitem__(self, name):
        """
        Direct access by name to the list of ions.
        """
        return self.ions[name]


    def __iter__(self):
        """
        Iterate over ions.
        """
        for ion in self.ions.values():
            yield ion


    def addIon(self, name, Z, n):
        """
        Add a new ion species.

        :param name: Ion species name.
        :param Z:    Ion species charge number.
        :param n:    Ion species total density.
        """
        if name in self.ions:
            raise Exception(f"Ion species with name '{name}' already exists.")

        self.ions[name] = IonSpecies(name=name, Z=Z, n=n)


    def cacheImpactIonizationRate(self, fre,):
        """
        Evaluate the fast-electron impact ionization rate for all species
        for the given distribution function.
        """
        for name in self.ions:
            self.ions[name].cacheImpactIonizationRate(fre=fre)


    def getElectronDensity(self):
        """
        Evaluates the electron density corresponding to the last
        ion rate equation solution.
        """
        ne = 0
        for ion in self.ions.values():
            ne += ion.getElectronDensity()

        return ne


    def getTotalElectronDensity(self):
        """
        Evaluate the total electron density (bound+free electrons)
        in the plasma represented by this IonHandler.
        """
        ne = 0
        for ion in self.ions.values():
            ne += sum(ion.solution) * ion.Z

        return ne


    def getLastSolution(self):
        """
        Get most recent solution to the system of equations.
        """
        N = self.getNumberOfStates()
        n = np.zeros((N,))

        offs = 0
        for ion in self.ions.values():
            n[offs:(offs+ion.Z+1)] = ion.solution

        return n


    def getNumberOfStates(self):
        """
        Count the number of charge states of all ions.
        """
        N = 0
        for ion in self.ions.values():
            N += ion.Z+1

        return N


    def getSolution(self):
        """
        Get the solution vector for all ion species.
        """
        N = self.getNumberOfStates()
        n = np.zeros((N,))

        offs = 0
        for ion in self.ions.values():
            n[offs:(offs+ion.Z+1)] = ion.solution
            offs += ion.Z+1

        return n


    def getNeutralDensity(self):
        """
        Get the neutral atom density of all species combined.
        """
        n = 0
        for ion in self.ions.values():
            n += ion.getNeutralDensity()

        return n


    def getTotalIonDensity(self, name=None):
        """
        Get the total ion density, of all or named species.
        """
        if name is None:
            n = 0
            for ion in self.ions.values():
                n += ion.n
            return n
        else:
            return self.ions[name].n


    def setDensity(self, name, n):
        """
        Set the total density of the named ion species.
        """
        self.ions[name].setDensity(n)


    def setSolution(self, n):
        """
        Set the ion densities from a solution.
        """
        offs = 0
        for ion in self.ions.values():
            ion.setSolution(n[offs:(offs+ion.Z+1)])
            offs += ion.Z+1
