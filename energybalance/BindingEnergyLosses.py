# Binding energy losses (ionization/recombination)

from scipy.constants import e


def eval(ion, Te, ne, Z0, **kwargs):
    """
    Evaluate the binding energy gain and losses (ionization and recombination)
    at the given electron temperature and density.
    """
    P = 0
    if Z0 > 0:
        dWi = e * ion.Wioniz[Z0-1]
        # Recombination gain
        P += dWi * ion.acd(Z0=Z0, n=ne, T=Te)

    if Z0 < ion.Z:
        dWi = e * ion.Wioniz[Z0]
        # Ionization loss
        P -= dWi * ion.scd(Z0=Z0, n=ne, T=Te)

    return ne*ion.solution[Z0] * P


def eval_dTe(ion, Te, ne, Z0, **kwargs):
    """
    Evaluate d(dW/dt)/dTe for the binding energy gain and loss term.
    """
    dP = 0

    if Z0 > 0:
        dWi = e * ion.Wioniz[Z0-1]
        P += dWi * ion.acd.deriv_Te(Z0=Z0, n=ne, T=Te)

    if Z0 < ion.Z:
        dWi = e * ion.Wioniz[Z0]
        P -= dWi * ion.scd.deriv_Te(Z0=Z0, n=ne, T=Te)

    return ne*ion.solution[Z0] * P


