# Routines for calculating energy balance
#
# MECHANISMS IMPLEMENTED
# Electron energy balance
#   [ ] Ohmic heating
#   [X] Line radiation
#   [X] Recombination radiation and bremsstrahlung
#   [X] Ionization radiation
#   [X] Runaway electron heating
#   [ ] Electron-ion collisions
# 
# Ion energy balance
#   [ ] Charge-exchange
#   [ ] Ion-electron, ion-ion & ion-neutralcollisions
#
# Neutral energy balance
#   [ ] Charge exchange
#   [ ] Ion-neutral collisions
#   [ ] Conduction
#   

from scipy.constants import e
from scipy.constants import k as k_B
import ionrate
import matplotlib.pyplot as plt
import numpy as np
import time

from chargeexchange import get_cx_rate
from energybalance import BindingEnergyLosses, CollisionalExchangeTerm, LineRadiation, PRBLosses, RunawayCollisions


# Wall temperature (in kelvin)
TWALL = 300


def drawtext(Te0, Te, v, label, color):
    intp = np.interp(Te0, Te, np.abs(v))
    plt.text(Te0, intp*2, label, color=color, fontsize=12)


def plot_Te_dependence(ions, fre, with_legend=False):
    """
    Plot the residual as a function of electron temperature.
    """
    Te = np.linspace(1, 10, 40)

    Terms = {}
    F = np.zeros(Te.shape)
    Tn = np.zeros((Te.size, len(ions)))
    CXrate = np.zeros((Te.size, len(ions)))
    Dnrate = np.zeros((Te.size, len(ions)))
    ierate = np.zeros((Te.size, len(ions)))
    start = time.time()
    for i in range(Te.size):
        F[i], terms, tn, cx, dn, ie = evaluate_residual(ions, fre, Te=Te[i])
        Tn[i,:] = tn
        CXrate[i,:] = cx
        Dnrate[i,:] = dn
        ierate[i,:] = ie

        for k, v in terms.items():
            if k not in Terms:
                Terms[k] = [float(v)]
            else:
                Terms[k].append(float(v))

    print(f'Evaluated residuals in {time.time()-start:.3f}s')

    # Locate sign changes
    T0 = []
    for i in range(1,F.size):
        if F[i-1]*F[i] < 0:
            T0.append(Te[i-1] - F[i-1] * (Te[i]-Te[i-1]) / (F[i]-F[i-1]))

    print('Equation has roots in')
    for i in range(len(T0)):
        print(f' --> Te = {T0[i]:.3f} eV')

    lbl1, lbl2 = 'Total', None
    if not with_legend:
        lbl1 = 'Energy GAIN'
        lbl2 = 'Energy LOSS'
        
    plt.figure(figsize=(6, 5))
    plt.semilogy(Te, F, 'k-', label=lbl1)
    plt.semilogy(Te, -F, 'k--', label=lbl2)

    if not with_legend:
        drawtext(Te0=2.5, Te=Te, v=F, label='Total', color='k')

    colors = ['tab:blue', 'tab:green', 'r', 'tab:orange', 'tab:purple', 'tab:brown']
    i = 0
    for k, v in Terms.items():
        lbl1, lbl2 = None, None
        if with_legend:
            lbl1 = k

        plt.plot(Te, v, label=lbl1, color=colors[i])
        plt.plot(Te, -np.array(v), '--', label=lbl2, color=colors[i])

        if not with_legend:
            Te0 = 7
            if i == 0: Te0 = 1.5

            drawtext(Te0=7, Te=Te, v=v, label=k, color=colors[i])

        i += 1

    plt.xlabel('Electron temperature (eV)')
    plt.ylabel('Residual (W/m$^{3}$)')
    plt.legend(loc='lower center', bbox_to_anchor=(.5,1), frameon=False, ncol=2)
    plt.tight_layout()

    plt.figure(figsize=(6, 4))
    Tn = np.array(Tn)
    i = 0
    for ion in ions:
        plt.semilogy(Te, Tn[:,i], label=ion.name)
        i += 1

    plt.plot(Te, Te, 'k--')
    plt.plot(Te, k_B/e * TWALL*np.ones(Te.shape), 'k--')
    plt.text(2, 3, r'$T_e$', fontsize=18)
    plt.text(3, 0.03, r'$T_{\rm wall}$', fontsize=18)
    plt.title('Neutral temperature')

    plt.xlabel('Electron temperature (eV)')
    plt.ylabel('Neutral temperature (eV)')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(6, 4))
    plt.title('Energy exchange timescales')
    
    i = 0
    for ion in ions:
        plt.semilogy(Te, CXrate[:,i], '-', color=colors[i+1], label=f'{ion.name}, CX')
        plt.semilogy(Te, Dnrate[:,i], '--', color=colors[i+1], label=f'{ion.name}, Dn')
        plt.semilogy(Te, ierate[:,i], ':', color=colors[i+1], label=f'{ion.name}, i-e')

        if i == 0:
            plt.semilogy([min(Te), max(Te)], [500e-3, 500e-3], 'k:', label=f'$500\,$ms')

        i += 1
    
    plt.plot([2.5, 2.5], [1e-6, 1e6], 'k--')
    plt.ylim([1e-5, 9e5])
    plt.xlabel('Electron temperature (eV)')
    plt.ylabel('Time scale (s)')
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    plt.show()


def find_equilibrium(ions, fre):
    """
    Find the equilibrium temperature with the given ions and RE distribution.
    """
    pass


def evaluate_residual(ions, fre, Te):
    """
    Evaluate the energy balance residual.
    """
    # Evaluate the particle balance
    #ionrate.equilibrium(ions, Te=Te, fre=fre)
    ionrate.equilibrium(ions, Te=Te, fre=None)

    terms = {
        'Binding energy': 0,
        'Line radiation': 0,
        'PRB losses': 0,
        'Ion-electron collisions': 0,
        'Neutral conduction': 0,
        'RE collisions': 0
    }

    # Iterate over ion species and charge states
    dWe = 0
    ne = ions.getElectronDensity()
    Tns = []
    CXrates = []
    Dnrates = []
    ierates = []
    for ion in ions:
        kwargs = dict(ion=ion, Te=Te, ne=ne)
        losses = 0
        for Z0 in range(0, ion.Z+1):
            kwargs['Z0'] = Z0
            #losses += BindingEnergyLosses.eval(**kwargs)
            #losses += LineRadiation.eval(**kwargs)
            #losses += PRBLosses.eval(**kwargs)
            terms['Binding energy'] += BindingEnergyLosses.eval(**kwargs)
            terms['Line radiation'] += LineRadiation.eval(**kwargs)
            terms['PRB losses']     += PRBLosses.eval(**kwargs)

        Ti = 4
        n0 = ion.solution[0] / ion.n
        #Tn = Te * (1 - 1.3*n0/(1+n0))
        #Tn = Te - n0*(Te-k_B/e*300)
        Di = 1000
        drw = 0.12

        for ion2 in ions:
            if ion2.name == ion.name:
                continue

            alpha = Di/drw**2 / (get_cx_rate(ion, ion2, ne, Ti) + Di/drw**2)
            CXrates.append((1/e)*(1/(get_cx_rate(ion, ion2, ne, Ti)*ion.solution[0]))[0])

        Twall = k_B/e * TWALL
        Tn = (1-alpha)*Te + alpha*Twall
        Tns.append(Tn[0])

        iDnrate = Di/drw**2 * ion.solution[0]
        terms['Neutral conduction'] += iDnrate * e*(Twall - Tn)
        Dnrates.append(1/(iDnrate*e))

        #ierates.append(1/CollisionalExchangeTerm.get_prefac(ion1=None, ion2=ion, T1=Te, T2=Ti, ne=ne, Te=Te)[0])
        ierates.append(np.abs((Te-Ti)/CollisionalExchangeTerm.eval(ion1=None, ion2=ion, T1=Te, T2=Ti, ne=ne, Te=Te)))
        #terms['Ion-electron collisions'] += CollisionalExchangeTerm.eval(ion1=None, ion2=ion, T1=Te, T2=Ti, ne=ne, Te=Te)
        terms['RE collisions'] += RunawayCollisions.eval(fre=fre, ne=ne, Te=Te)

        #dWe += losses + gains

    dWe = 0
    for v in terms.values():
        dWe += v

    F = np.zeros((len(ions),))

    F[0] = dWe

    return dWe, terms, Tns, CXrates, Dnrates, ierates


