__all__ = [
    'GRAV', 'AVO', 'RGASUNIV', 'STDATMPA', 'STDTEMP', 'MWAIR',
    'du2mm2', 'du2cm2'
]

import scipy.constants

GRAV = scipy.constants.value('standard acceleration of gravity')
AVO = scipy.constants.value('Avogadro constant')
RGASUNIV = scipy.constants.value('molar gas constant')

# standard atmosphere  [ Pa ]
STDATMPA = 101325.0
# Standard Temperature [ K ]
STDTEMP = 273.15

# CMAQ
# mean molecular weight for dry air [ g/mol ]
# FSB: 78.06% N2, 21% O2, and 0.943% A on a mole
# fraction basis ( Source : Hobbs, 1995) pp. 69-70
# MWAIR = 28.9628 / 1e3
#
# Adopted in kg/mole
MWAIR = 28.9628 / 1e3


def du2mm2(dobsons):
    """
    Conversion from Dobson units to molecules/m2
    """
    mm2 = dobsons * STDATMPA * AVO / RGASUNIV / STDTEMP * 0.01e-3
    return mm2


def du2cm2(dobsons):
    """
    Conversion from Dobson units to molecules/cm2
    """
    return du2mm2 / 1e4
