# Term representing energy exchange via collisions
# (eq. (44) in the DREAM paper)

import numpy as np
from scipy.constants import e, epsilon_0, m_e, m_p


def get_prefac(ion1, ion2, ne, Te, **kwargs):
    """
    Returns the pre-factor for this term (which does NOT depend on the particle
    temperatures T1 and T2).
    """
    avnZ1, avnZ2 = 0, 0

    # Is species 1 electrons?
    if ion1 is None:
        avnZ1 = ne
        Z1 = 1
        m1 = m_e
    else:
        Z1 = ion1.Z
        m1 = 2*Z1 * m_p
        for Z0 in range(1, ion1.Z+1):
            avnZ1 += ion1.solution[Z0] * Z0**2

    # Is species 2 electrons?
    if ion2 is None:
        avnZ2 = ne
        Z2 = 1
        m2 = m_e
    else:
        Z2 = ion2.Z
        m2 = 2*Z2 * m_p
        for Z0 in range(1, ion2.Z+1):
            avnZ2 += ion2.solution[Z0] * Z0**2

    if ion1 is None or ion2 is None:
        # e-i Coulomb logarithm
        lnLambda = 14.9 - 0.5*np.log(ne/1e20) + np.log(Te/1000)
    else:
        # i-i Coulomb logarithm
        lnLambda = 17.3 - 0.5*np.log(ne/1e20) + 1.5*np.log(Te/1000)

    prefac = avnZ1*avnZ2 * e**4 * lnLambda / ((2*np.pi)**(3/2) * epsilon_0**2 * m1*m2)

    return prefac, m1, m2


def eval(ion1, ion2, T1, T2, ne, Te, **kwargs):
    """
    Evaluate the energy exchange between two ion species, 1 & 2, with
    temperatures T1 and T2 respectively.
    """
    prefac, m1, m2 = get_prefac(ion1=ion1, ion2=ion2, ne=ne, Te=Te)
    Tfac = (e*T2-e*T1) / (e*T1/m1 + e*T2/m2)**(3/2)

    return prefac * Tfac


def eval_dT1(ion1, ion2, T1, T2, **kwargs):
    """
    Evaluate the derivative of the collisional exchange term w.r.t. to T1.
    """
    prefac, m1, m2 = get_prefac(ion1=ion1, ion2=ion2, ne=ne, Te=Te)
    Tfac = ((1/3)*e*T1-e*T2-(2/3)*(m1/m2)*T2) / (m1*(e*T1/m1 + e*T2/m2)**(5/2))

    return prefac*Tfac


def eval_dT2(ion1, ion2, T1, T2, **kwargs):
    """
    Evaluate the derivative of the collisional exchange term w.r.t. to T2.
    """
    prefac, m1, m2 = get_prefac(ion1=ion1, ion2=ion2, ne=ne, Te=Te)
    Tfac = (e*T1+(2/3)*(m2/m1)*T1 - (1/3)*e*T2) / (m2*(e*T1/m1 + e*T2/m2)**(5/2))

    return prefac * Tfac


def eval_rate(ion1, ion2, T1, T2):
    """
    Evaluate the energy transfer rate.
    """
    prefac, m1, m2 = get_prefac(ion1=ion1, ion2=ion2, ne=ne, Te=Te)
    Tfac = 1 / (e*T1/m1 + e*T2/m2)**(3/2)

    return prefac*Tfac


