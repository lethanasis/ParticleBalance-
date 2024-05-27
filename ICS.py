# Implementation of the relativistic electron ionization cross-section (ICS).

import numpy as np
from scipy.constants import alpha, c, e, h, m_e, physical_constants, Rydberg


def evaluate(p, C, I_pot_eV, betaStar):
    """
    Evaluate the ionization cross-section (ICS) for a relativistic electron
    colliding with a Maxwellian, according to the Garland model.
    """
    mc2 = m_e*c**2
    Ry = Rydberg * h*c / e        # Rydberg energy in eV
    a0 = physical_constants['Bohr radius'][0]
    # Convert to units of mc^2
    I_pot = I_pot_eV / (mc2/e)
    gamma = np.sqrt(1+p**2)
    # Electron kinetic energy in units of mc^2
    E = p**2/(1+gamma)
    beta = p/gamma

    # Electron kinetic energy per ionization threshold energy
    U = E/I_pot

    if U < 1:
        return 0

    prefac = C*np.pi*a0**2

    sigmaNR = prefac * Ry**2 / I_pot_eV**2 * np.power(np.log(U), 1+betaStar/U) / U
    sigmaR  = prefac * alpha**2 * Ry / I_pot_eV * (np.log(p**2/(2*I_pot)) - beta**2)

    S = 1/(1+np.exp(1-E*mc2/e * 1e-5))

    return (1-S)*sigmaNR + S*sigmaR


def evaluate_ioniz_rate(p, C, I_pot_eV, betaStar, f_re):
    """
    Evaluate the ionization rate for the given distribution function.
    """


