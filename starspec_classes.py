import numpy as np

_ATTR_NAMES = ('teffgrid','logggrid','metalgrid','grid','rv_model','bestparams')
class modelgrid:
    def __init__(self):
        for key, value in zip(_ATTR_NAMES, np.zeros(len(_ATTR_NAMES))):
            setattr(self, key, value)

class parameters:
    def __init__(self):
        self.teff = None
        self.logg = None
        self.metal = None
        self.fbol = None
        self.teff_err = None
        self.logg_err = None
        self.metal_err = None
        self.fbol_err = None

class spectrum:
    def __init__(self):
        self.wave = None
        self.s = None
        self.s_err = None

class photometry:
    def __init__(self):
        self.system = None
        self.bandpass = None
        self.mag = None
        self.err = None
        
class starinfo:
    def __init__(self):
        self.parameters = parameters()
        self.spectrum = spectrum()
        self.photometry = None

