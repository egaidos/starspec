# make a pickle file from a table of photometry

import numpy as np
import pickle
import pandas as pd
from astropy.io import ascii

from starspec_classes import *

filename = '/Users/gaidos/Desktop/WORK/K2/DIPPERS/EPIC2481/PHOTOMETRY/epic2481_photo.txt'
starname = "EPIC 248131102"
photometryfile = 'test.pkl'
f = ascii.read(filename)
systems = np.array(f['col1'])
bands = np.array(f['col2'])
mags = np.array(f['col3'])
mag_errs = np.array(f['col4'])

df = pd.DataFrame()

photodata = dict()
observations = []

for system, band, mag, mag_err in zip(systems,bands,mags,mag_errs):
    observation = photometry()
    observation.system = system
    observation.bandpass = band
    observation.mag = mag
    observation.mag_err = mag_err
    observations.append(observation)

photodata[starname] = observations

with open(photometryfile,'wb') as output:
    pickle.dump(photodata,output,pickle.HIGHEST_PROTOCOL)
    
