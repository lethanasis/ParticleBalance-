# Recombination radiation and bremsstrahlung losses


def eval(ion, Te, ne, Z0, **kwargs):
    """
    Evaluate the recombination radiation and bremsstrahlung losses for the given
    ion species, at the given electron temperature and density.
    """
    if Z0 > 0:
        return -ne * ion.solution[Z0] * ion.prb(Z0=Z0, n=ne, T=Te)
    else:
        return 0


def eval_dTe(ion, Te, ne, Z0, **kwargs):
    """
    Evaluate the derivative of the recombination radiation and bremsstrahlung
    losses w.r.t. the electron temperature.
    """
    if Z0 > 0:
        return -ne * ion.solution[Z0] * ion.prb.deriv_Te(Z0=Z0, n=ne, T=Te)
    else:
        return 0


