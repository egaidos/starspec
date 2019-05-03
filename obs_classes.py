class spex:
    def __init__(self):
        self.wave = None
        self.flux = None
        self.err = None

class snifsblue:
    def __init__(self):
        self.flux = None
        self.wave = None
        self.err = None
        self.image = None

class snifsred:
    def __init__(self):
        self.flux = None
        self.wave = None
        self.err = None
        self.image = None

class snifs:
    def __init__(self):
        self.header = None
        self.blue = snifsblue()
        self.red = snifsred()

class photometry:
    def __init__(self):
        self.system = None
        self.bandpass = None
        self.mag = None
        self.err = None

class starobs:
    def __init__(self):
        self.snifs = snifs()
        self.spex = spex()
        self.photometry = photometry()



