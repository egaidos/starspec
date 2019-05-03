import numpy as np
import pickle
import pandas as pd
import argparse

p = argparse.ArgumentParser()
p.add_argument("starname",nargs=1)
args = p.parse_args()
starname = args.starname
starname = str(starname[0])

catalogfile = 'uvm_catalog.pkl'
cat = pd.read_pickle(catalogfile)
teff = cat[starname].parameters.teff
teff_err = cat[starname].parameters.teff_err
logg = cat[starname].parameters.logg
logg_err = cat[starname].parameters.logg_err
fbol = cat[starname].parameters.fbol
fbolerr = cat[starname].parameters.fbol_err
print('Fbol = ',fbol,fbolerr)
print('Teff = ',teff,teff_err)
print('log g = ',logg,logg_err)
print('[Fe/h] = ',cat[starname].parameters.irmetal,cat[starname].parameters.irmetal_err)
