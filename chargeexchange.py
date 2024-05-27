# Routines for charge-exchange-related calculations


def get_cx_rate(ion1, ion2, ne, Ti):
    """
    Evaluate the charge-exchange rate for two ion species.
    """

    if ion1.name == 'D':
        Rcx = ion2.ccd(Z0=1, n=ne, T=Ti)
    else:
        Rcx = ion1.ccd(Z0=1, n=ne, T=Ti)

    return 3/2 * Rcx * ion2.solution[1]


