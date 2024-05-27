# Line radiation losses


def eval(ion, Te, ne, Z0, **kwargs):
    """
    Evaluate the line radiation losses for the given ion species, at the
    given electron temperature and density.
    """
    if Z0 < ion.Z:
        return -ne * ion.solution[Z0] * ion.plt(Z0=Z0, n=ne, T=Te)
    else:
        return 0


def eval_dTe(ion, Te, ne, Z0, **kwargs):
    """
    Evaluate the derivative of the line radiation losses w.r.t. the
    electron temperature.
    """
    if Z0 < ion.Z:
        return -ne * ion.solution[Z0] * ion.plt.deriv_Te(Z0=Z0, n=ne, T=Te)
    else:
        return 0


