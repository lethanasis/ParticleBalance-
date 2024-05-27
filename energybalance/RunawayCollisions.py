# Term describing collisional energy transfer from runaways
# to thermal electrons

import numpy as np
from scipy.constants import c, e, epsilon_0, m_e


def eval(fre, ne, Te, deriv=False):
    """
    Evaluate the power transferred from a distribution of runaway electrons
    to thermal electrons of the given density.
    """
    p = fre.p
    v = c * p / np.sqrt(1+p**2)
    f = 4*np.pi * np.sum(fre.f.T * fre.dxi, axis=1)

    Ffr = friction_force(p, ne, Te)
    I = np.sum(f * Ffr * v * p**2 * fre.dp)

    return I


def eval_dT(fre, ne, Te):
    """
    Evaluate the derivative of this term w.r.t. the electron temperature.
    """
    return eval(fre=fre, ne=ne, Te=Te, deriv=True)


def coulombLogarithm(p, ne, Te, deriv=False):
    """
    Evaluates the e-e Coulomb logarithm for an electron with momentum p
    colliding against a Maxwellian electron population with temperature Te
    and density ne.
    """
    k = 5
    if deriv:
        g = np.sqrt(1+p**2)
        corr_dT = -0.5 * (g-1)**(k/2) / (1 + (g-1)/(e*Te/(m_e*c**2))) / (e*Te/(m_e*c**2))**(k/2) / (e*Te)
        ln0_dT = 1000/Te

        return ln0_dT + corr
    else:
        corr = 1/k * np.log(1 + ((np.sqrt(1+p**2)-1)/(e*Te/(m_e*c**2)))**(k/2))
        ln0  = 14.9 + np.log(Te/1000) - 0.5*np.log(ne/1e20)

        return ln0 + corr


def friction_force(p, ne, Te, deriv=False):
    """
    Evaluate the collisional friction force experienced by an electron
    of momentum p against a background Maxwellian population of electrons
    of density ne.
    """
    logLambda = coulombLogarithm(p, ne, Te, deriv=deriv)
    v = c * p / np.sqrt(1+p**2)
    F = e**4 * ne * logLambda / (4*np.pi*epsilon_0**2*m_e*v**2)

    return F
    

