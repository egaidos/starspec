# make a spectrum file

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from astropy.io import ascii, fits
import os

from starspec_classes import *
from obs_classes import *


#obs.snifs.header = header
#obs.snifs.blue.wave = bluewave
#obs.snifs.blue.flux = blueflux
#obs.snifs.blue.image = blueimage
#obs.snifs.red.wave = redwave
#obs.snifs.red.flux = redflux
#obs.snifs.red.image = redimage


snifsfile = '/Users/gaidos/Desktop/WORK/K2/AGOG/SNIFSCAL/test.pkl'
observations = pd.read_pickle(snifsfile)
observations["EPIC 248131102"] = observations['EPIC_2481']


spexfile = '/Users/gaidos/Desktop/WORK/K2/DIPPERS/EPIC2481/SpeX/EPIC_248131102_tellcor_merged.txt'
f = ascii.read(spexfile)
wave = np.array(f['col1'])
flux = np.array(f['col2'])
flux_err = np.array(f['col3'])

x = observations["EPIC 248131102"].snifs.blue.flux
y = observations["EPIC 248131102"].snifs.blue.err

observations["EPIC 248131102"].spex.wave = wave
observations["EPIC 248131102"].spex.flux = flux
observations["EPIC 248131102"].spex.err = flux_err
print(observations["EPIC 248131102"].snifs.blue.wave)

with open('epic2481_spec.pkl','wb') as output:
    pickle.dump(observations,output,pickle.HIGHEST_PROTOCOL)

