#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from DREAM import DREAMSettings, runiface
import DREAM.Settings.Equations.IonSpecies as Ions
import DREAM.Settings.Equations.ColdElectronTemperature as Temperature
import DREAM.Settings.Solver as Solver
import DREAM.Settings.Atomics as Atomics
import sys
from scipy.constants import c

sys.path.append('..')
import config
from DistributionFunction import DistributionFunction
from IonHandler import IonHandler
import ionrate


def generate_settings(T0, nD, nImp, Zimp, fre):
    """
    Generate DREAM settings.
    """
    ds = DREAMSettings()

    ds.atomic.adas_interpolation = Atomics.ADAS_INTERP_BILINEAR

    ds.hottailgrid.setEnabled(False)
    ds.runawaygrid.setEnabled(False)

    # Radial grid
    ds.radialgrid.setNr(1)
    ds.radialgrid.setB0(1.45)
    ds.radialgrid.setMinorRadius(0.25)
    ds.radialgrid.setWallRadius(0.30)

    # Background plasma
    ds.eqsys.E_field.setPrescribedData(0)
    ds.eqsys.T_cold.setPrescribedData(T0)

    imp = {10: 'Ne', 18: 'Ar', 36: 'Kr'}
    ds.eqsys.n_i.addIon(name='D', Z=1, iontype=Ions.IONS_DYNAMIC, init_equil=True, n=nD)
    if nImp > 0:
        ds.eqsys.n_i.addIon(name=imp[Zimp], Z=Zimp, iontype=Ions.IONS_DYNAMIC, init_equil=True, n=nImp)
        #ds.eqsys.n_i.addIon(name='Ar', Z=18, iontype=Ions.IONS_DYNAMIC, init_equil=True, n=nImp)

    # Include runaways?
    if fre:
        ds.runawaygrid.setEnabled(True)
        ds.eqsys.n_i.setIonization(Ions.IONIZATION_MODE_KINETIC)

        ds.runawaygrid.setNp(fre.p.size)
        ds.runawaygrid.setNxi(fre.xi.size)
        ds.runawaygrid.setPmin(fre.pMin)
        ds.runawaygrid.setPmax(fre.pMax)

        f = np.zeros((1, 1, fre.xi.size, fre.p.size))
        f[0,0,:] = fre.f
        ds.eqsys.f_re.prescribe(f=f, t=[0], r=[0], p=fre.p, xi=fre.xi)

    ds.other.include('fluid', 'scalar')
    if fre is not None:
        ds.other.include('runaway/kinioniz_vsigma')

    ds.solver.setType(Solver.LINEAR_IMPLICIT)
    ds.timestep.setTmax(5e-1)
    ds.timestep.setNt(200)

    return ds


def compareICS(do, ions):
    """
    Compare the ionization cross-sections calculated by DREAM and the
    particle balance code.
    """
    p = do.grid.runaway.p[:]
    for ion in ions:
        plt.figure()
        plt.title(f'{ion.name}')
        for Z0 in range(ion.Z):
            sigma = np.zeros((p.size,))
            for i in range(p.size):
                sigma[i] = ion.evaluateICS(p[i], Z0)

            v = c*p/np.sqrt(1+p**2)
            #plt.plot(p, c*p/np.sqrt(1+p**2) * sigma, 'k')
            #plt.plot(p, do.other.runaway.kinioniz_vsigma[ion.name][Z0][-1,0,0,:], 'r--')

            # Relative error
            plt.plot(p, do.other.runaway.kinioniz_vsigma[ion.name][Z0][-1,0,0,:] / (v*sigma) - 1, label=f'$Z_0 = {Z0}$')

        plt.xlabel('$p/mc$')
        plt.ylabel(r'$\left\langle v\sigma \right\rangle$')
        plt.legend()
        break

    plt.show()


def main():
    ions = IonHandler()

    #nD = 1e19
    #nI = 4e18
    #ZI = 18
    #Te = 2.1
    # 75838
    nD = 7.200070891065e+19
    nI = 1.554404145078e+18
    #ZI = 10
    ZI = 18
    Te = 1.0712

    ions.addIon('D', Z=1, n=nD)
    #ions.addIon('Ne', Z=10, n=nI)
    ions.addIon('Ar', Z=18, n=nI)

    fre = DistributionFunction()
    fre.setStep(nre=1e16, pMin=1, pMax=60, pUpper=60, nP=1000)
    #fre.setAvalanche(nre=1e16, pMin=1, pMax=20)
    #fre = None

    ds = generate_settings(T0=Te, nD=nD, nImp=nI, Zimp=ZI, fre=fre)
    ds.save('settings_benchmark.h5')
    do = runiface(ds, 'output_benchmark.h5')

    n = ionrate.equilibrium(ions, Te=Te, fre=fre)

    if fre:
        print(f'p0:         {fre.p[0]}')
        print(f'pMax:       {fre.p[-1]}')
        print(f'DREAM p0:   {do.grid.runaway.p[0]}')
        print(f'DREAM pMax: {do.grid.runaway.p[-1]}')

    print('       n             DREAM       Delta')
    i = 0
    for ion in ions:
        N = sum(n[i:(i+ion.Z+1)])
        print(f':: {ion.name}')
        print(f'     {N:12.4e}  {sum(do.eqsys.n_i[ion.name].data[-1,:,0]):12.4e}')
        for Z0 in range(ion.Z+1):
            Dn = do.eqsys.n_i[ion.name][Z0][-1,0]
            print(f'{Z0:2d}:  {n[i]:12.4e}  {Dn:12.4e}  {abs(n[i]-Dn)/N*100:.3f}%')

            i += 1

    #compareICS(do, ions)

    return 0


if __name__ == '__main__':
    sys.exit(main())


