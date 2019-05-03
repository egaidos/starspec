#  spectrophometric fitting of combined SNIFS+SpeX spectra

# working flux densities are in ergs/s/cm2/Angstrom
# working wavelengths are in *microns*
# ha! mixed units

# call:   python3 starspec.py STARNAME --exclude FILTERS
# FILTERS is a list of filters separated by '-'

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from astropy.io import fits
import pickle
from scipy.special import gammainc
import argparse
from pathlib import Path

from starspec_classes import *
from starspec_procedures import *

def main():

    p = argparse.ArgumentParser()
    p.add_argument("starname",nargs=1)
    p.add_argument('--grid',dest='gridfile')
    p.add_argument('--filter',dest='filterfile')
    p.add_argument('--spectra',dest='specfile')
    p.add_argument('--photometry',dest='photofile')
    p.add_argument('--model_iter',dest='model_iter')
    p.add_argument('--photo_iter',dest='photo_iter')
    p.add_argument('--loop_iter',dest='loopiter')
    p.add_argument('--snr_min',dest='snrmin') # replace spectrum with model if below this value (not yet implemented?)
    p.add_argument('--n_models',dest='nbest') # number of models around which to build regular grid
    p.add_argument('--wave_max',dest='wave_max') # maximum wavelength out which to append spectrum for fitting
    p.add_argument('--n_append',dest='nappend') # number of points in appendix
    p.add_argument('--makegrid',action='store_true') # (re)make the grid
    p.add_argument('--exclude',dest='exclude') # exclude one or more filters by name, separated by commas
    p.add_argument('--newphoto',dest='newphoto', action='store_true')
    p.set_defaults(newphoto=False) # incorporate photometry from the database
    p.add_argument('--ebv',dest='ebv',nargs=2) # reddening and its error

    args = p.parse_args()
    starname = args.starname
    starname = str(starname[0])
    gridfile = args.gridfile
    filterfile = args.filterfile
    specfile = args.specfile
    photofile = args.photofile
    model_iter = args.model_iter
    photo_iter = args.photo_iter
    loopiter = args.loopiter
    nbest = args.nbest
    wave_max = args.wave_max
    nappend = args.nappend
    exclude = args.exclude
    newphoto = args.newphoto
    ebv = args.ebv # 
    

    # default values
    if gridfile == None: gridfile = '../Grid/phoenix_aces_USEME.fits' #'../Grid/PHOENIX-BTSETTL-CIFIST_NEW.fits'
    if filterfile == None: filterfile = 'filter.txt' # file containing problematic wavelength ranges to replace with model
    if specfile == None: specfile =  '../Data/uvm_spec_new.h5' #'../Data/ld_spec.h5' # database for spectroscopy
    if photofile == None: photofile = '../PHOTOMETRY/photometry_uvm_20180703.pkl'   # '../PHOTOMETRY/photometry_LD.pkl' # database for photometry
    if loopiter == None: loopiter = 10 # number of over-all fitting loops
    if model_iter == None: model_iter = 10 # number of spectral model loops in big loop 
    if photo_iter == None: photo_iter = 20 # number of photometry fitting loops in big loop
    if nbest == None: nbest = 27 # number of best-fit models to select to build subgrid
    if wave_max == None: wave_max = 6.0 # wavelength to extend to do photometry
    if nappend == None: nappend = 3000
    if ebv == None: 
        ebv = 0. # default is no extinction 
        ebv_err = 0.
    else:
        ebv_err = float(ebv[1])
        ebv = float(ebv[0])

    preiter = 50 # photometric fitting iterations before sub-grid is build

    err_iter = 30 # number of Monte Carlos to generate errors
    catalogfile = 'test.pkl' #'uvm_catalog.pkl'

    photohashfile = '../PHOTOMETRY/photohash.txt'
    photsysfile = '../PHOTOMETRY/phot_systems.dat'
    waverange = [1.165,1.295] # wavelength range over which to determine spectral resolution

    #av = 3.1*ebv  # EXPERIMENTAL

    extension = specfile.rpartition(".")[-1]

    if extension == "h5":
        f = pd.read_hdf('../Data/'+specfile,'table')
        starnames = np.array(f['#'])
        indices = np.arange(len(starnames))
        nstar = np.asscalar(indices.compress(starnames == starname))
    else:
        nstar = -1

    # build target spectrum
    wave, flux, err, wave_splice_snifs, wave_splice_spex = build_spectrum(specfile,starname,nstar)
    flux = flux[np.argsort(wave)]
    err = err[np.argsort(wave)]
    wave = wave[np.argsort(wave)]
    which = (err/flux < 0.1)
    #plt.plot(wave[which],flux[which],'black')
    #plt.ylim([0,2*np.median(flux)])
    #plt.xlim([0.4,2.5])
    #plt.xscale('log')
    #plt.xlabel('microns')
    #plt.savefig('epic2481spec.png')
    #plt.show(block=True)


    #plt.plot(wave,flux)
    #plt.pause(2)

    print('Splice points = ', wave_splice_snifs,wave_splice_spex)
    if (wave_splice_snifs == 0.):
        print('Spectrum of ',starname,' incomplete!')
    else:
        print('Analyzing spectrum of ',starname)
        subgridfile = '../Grid/'+starname+'_grid.pkl' # comparison subgrid
        plotfile = starname+'_plot.png' 

        snr = flux/err
        filter = build_filter(wave,filterfile) # build filter
        res, wave_rv, flux_rv  = resolution(wave,flux,err,waverange,filter) # find resolution of spex
        # and extract 'rv' section for x-correlation purposes
        if newphoto == True:
            systems, bands, mag, mag_err = get_photometry(photofile,photohashfile,starname,exclude) # get available photometry
        else:
            catalog = pd.read_pickle(catalogfile)
            star = catalog[starname]
            observations = star.photometry
            systems = []
            bands = []
            mag = []
            mag_err = []
            for observation in observations:
                systems.append(observation.system)
                bands.append(observation.bandpass)
                mag.append(observation.mag)
                mag_err.append(observation.mag_err)
            print(mag_err)
        
        mag_synth = []
        mag_synth_err = []
        waveb = []
        extra_err = np.array(mag_err)*0

        rv_nom = 0.
        feh, feh_err, feh_j, feh_h, feh_k  = metallicity(wave,flux,err,rv_nom) # don't use patched spectrum for this since some Fe/H features filtered
        
        print('Initial Metallicity = ', feh, '+/-',feh_err)

        ##### experimental ######   first does a photometric fit
        photparams = np.array([1.,1.,1.,1.]) # initial photometric adjustment parameters
        photo_result = minimize(fit_photometry,photparams,args=(wave_splice_snifs,wave_splice_spex,wave,flux,err,photsysfile,systems,bands,mag,mag_err,ebv),method='Nelder-Mead',options={'maxiter':preiter})
        photparams = photo_result.x
        fluxgrid, errgrid = warp_spec(photparams,wave_splice_snifs,wave_splice_spex,wave,flux,err) # alter the spectrum accordingly
        #######################

        filepath = Path(subgridfile)
        if filepath.is_file(): # if subgrid exists then read  in 
            s = pd.read_pickle(subgridfile)
            teffgrid = s.teffgrid
            logggrid = s.logggrid
            metalgrid = s.metalgrid
            grid = s.grid
            rv_model = s.rv_model
            bestparams = s.bestparams
        else: # otherwise create one

            #teffgrid, logggrid, metalgrid, grid, rv_model, bestparams  = build_grid(gridfile,nbest,res,wave,flux,err,filter,wave_rv,flux_rv,feh,feh_err)
            #teffgrid, logggrid, metalgrid, grid, rv_model, bestparams  = build_grid(gridfile,nbest,res,wave,fluxgrid,errgrid,filter,wave_rv,flux_rv,feh,feh_err)
            teffgrid, logggrid, metalgrid, grid, rv_model, bestparams  = build_grid(gridfile,nbest,res,wave,ebv,fluxgrid,errgrid,filter,wave_rv,flux_rv,feh,feh_err) # EXPERIMENTAL
            models = modelgrid()
            models.teffgrid = teffgrid
            models.logggrid = logggrid
            models.metalgrid = metalgrid
            models.grid = grid
            models.rv_model = rv_model
            models.bestparams = bestparams
            with open(subgridfile,'wb') as output:
                pickle.dump(models,output,pickle.HIGHEST_PROTOCOL)

        #bestparams = np.append(bestparams,av) # EXPERIMENTAL
        rv_nom = np.median(rv_model) # nominal RV to use to compute metallicity indices
        print('Nominal RV = ',-rv_nom)
        gridparams = [teffgrid,logggrid,metalgrid]

        bestparams[2] = feh
        bestfit_params = bestparams
        print('Initial best parameters = ',bestfit_params)
        # compute metallicity

        #feh, feh_err, feh_j, feh_h, feh_k  = metallicity(wave,flux,err,rv_nom) # don't use patched spectrum for this since some Fe/H features filtered
        #print('Metallicity = ', feh, '+/-',feh_err)

        isv = np.zeros((4,3)) # reasonable initial simplex
        isv[0,:] = bestparams+np.array([50., 0.0,  0.])
        isv[1,:] = bestparams+np.array([50., 0.25, 0.])
        isv[2,:] = bestparams+np.array([0.0, 0.25, 0.2])
        isv[3,:] = bestparams+np.array([0.0, 0.0,  0.2])
        #isv = np.zeros((5,4)) # reasonalbe initial simplex    
        #isv[0,:] = bestparams+np.array([50., 0., 0., 0.])
        #isv[1,:] = bestparams+np.array([50.,0.3, 0., 0.])
        #isv[2,:] = bestparams+np.array([0., 0.3, 0.2, 0.])
        #isv[3,:] = bestparams+np.array([0., 0., 0.2, 0.1])
        #isv[4,:] = bestparams+np.array([0.,0.0,0.,0.1])
        #photparams = np.array([1.,1.,1.,1.]) # initial photometric adjustment parameters
        photparams_best = photparams[:]

        for iter in np.arange(loopiter):  # loop to iterate between model and photometry fitting

            print('Iter = ',iter+1)

            # fit model to adjusted spectrum
            if (iter == 0):
                print(bestfit_params)
                print(isv)
                model_result = minimize(fit_spectrum,bestfit_params,args=(wave,flux,err,snr,wave_splice_snifs,wave_splice_spex,photparams,filter,gridparams,grid,feh,feh_err,ebv),method='Nelder-Mead',options={'maxiter':model_iter, 'initial_simplex':isv})
                chi2min = model_result.fun
            else:
                bestfit_params_try = bestfit_params[:]
                model_result_try = minimize(fit_spectrum,bestfit_params_try,args=(wave,fluxmod_short,errmod_short,snr,wave_splice_snifs,wave_splice_spex,photparams,filter,gridparams,grid,feh,feh_err,ebv),method='Nelder-Mead',options={'maxiter':model_iter})
                if (model_result_try.fun < chi2min):
                    chi2min = model_result_try.fun
                    model_result = model_result_try
                    bestfit_params = bestfit_params_try[:]
                    photparams_best = photparams

                else:  # if chi2 larger than minimum find which passband has most deviation and increase error to match offset
                    waveb = []
                    varmag = []
                    maxdev = 0.
                    for systemval, bandval, magval, mag_errval in zip(systems,bands,mag,mag_err): # generate synthetic magnitudes
                        transmission, zeropoint, waveband = build_bandpass(photsysfile,systemval,bandval,wave_whole)
                        magsynth, magsynth_err = synth_mag(wave_whole,fluxmod,errmod,transmission,zeropoint)
                        varmag.append(np.asscalar((magval - magsynth)**2 - mag_errval**2))
                    varmag = np.array(varmag) - extra_err**2
                    maxindex = np.argmax(varmag) #/(np.array(mag_err))**2)
                    if (varmag[maxindex] > 0.):
                        extra_err[maxindex] = np.sqrt(varmag[maxindex] + (extra_err[maxindex])**2)
                        print('add extra error to ', systems[maxindex],bands[maxindex],' of ',extra_err[maxindex])

            bestfit_params = model_result.x
            print('Stellar: ',bestfit_params,' chi-squared = ',model_result.fun)

            # patch and extend spectrum with best-fit model
            patched_spectrum, patched_err, ratio = patch_spectrum(bestfit_params,wave,flux,err,snr,filter,gridparams,grid)
            wave_append = np.array(max(wave) + (wave_max - max(wave))*np.arange(nappend)/(1.*nappend))
            flux_append, err_append = append_spectrum(gridfile,teffgrid,logggrid,metalgrid,rv_model,ratio,bestfit_params,wave_append)
            wave_whole = np.concatenate([wave,wave_append])
            flux_whole = np.concatenate([patched_spectrum,flux_append])
            err_whole = np.concatenate([patched_err,err_append])
            
            indices = np.argsort(wave_whole)
            wave_whole = wave_whole[indices]
            flux_whole = flux_whole[indices]
            err_whole = err_whole[indices]
            
            # fit spectrum to photometry

            mag_err_tot = np.sqrt(np.array(mag_err)**2 + extra_err**2)  # EXPERIMENTAL   add extra error in for photometric fitting
            photo_result = minimize(fit_photometry,photparams,args=(wave_splice_snifs,wave_splice_spex,wave_whole,flux_whole,err_whole,photsysfile,systems,bands,mag,mag_err_tot,ebv),method='Nelder-Mead',options={'maxiter':photo_iter})
            photparams = photo_result.x
            print('Photometry: ',photparams,' chi-squared = ',photo_result.fun)

            ####experimental#########   modifies spectrum according to photometry
            fluxmod, errmod = warp_spec(photparams,wave_splice_snifs,wave_splice_spex,wave_whole,flux_whole,err_whole) # alter the spectrum accordingly
            fluxmod_short = flux_whole[0:len(wave)] # unmodifieid spectrum to compare consistency between spectrum and photometry
            errmod_short = err_whole[0:len(wave)]

            ################

        # end of fitting master loop

        chisqmodel = model_result.fun
        chisqphoto = photo_result.fun

        fluxmod, errmod = warp_spec(photparams_best,wave_splice_snifs,wave_splice_spex,wave_whole,flux_whole,err_whole) # alter the spectrum accordingly

        fbol = fluxcalc(wave_whole,fluxmod) # calculate bolometric flux

        # Monte carlo for errors
        fbolmc = []
        teffmc = []
        loggmc = []
        metalmc = []
        #ebvmc = [] # EXPERIMENTAL

        # find value of additional erorr term that makes photometry reduced chi-squared of one
        print('finding systematic photometry error....')
        sys_err = 0.
        dof = len(mag) - 4
        chi2 = 1.0e6
        ntry = 0
        while ((chi2 > dof) & (ntry < 100)):
            mags_syn = []
            mags_syn_err = []
            waveb = []
            chi2 = 0.
            errval = np.sqrt((np.array(mag_err))**2 + sys_err**2)
            err_tot = errval.tolist()
            for systemval, bandval, magval, mag_errval in zip(systems,bands,mag,err_tot): # generate synthetic magnitudes
                transmission, zeropoint, waveband = build_bandpass(photsysfile,systemval,bandval,wave_whole)
                magsynth, magsynth_err = synth_mag(wave_whole,fluxmod,errmod,transmission,zeropoint)
                mags_syn.append(np.asscalar(magsynth))
                mags_syn_err.append(np.asscalar(magsynth_err))
                waveb.append(np.asscalar(waveband))
                chi2 = chi2 + ((magval - magsynth)**2/(mag_errval**2 + magsynth_err**2))
            sys_err = sys_err + 0.1*np.median(mag_err)
            ntry += 1
        print('systematic photometry error = ',sys_err)

        # Monte Carlo iteration to determine errors
        #tot_err = np.sqrt(np.array(mag_err)**2 + sys_err**2)
        tot_err = np.sqrt(np.array(mag_err)**2 + extra_err**2) # EXPERIMENTAL
        for iter in np.arange(err_iter):
            print('Monte Carlo iteration ',iter+1)
            mc_modelparams = bestfit_params[:]
            mc_photoparams = photparams[:]
            ebvmc = ebv + ebv_err*np.random.normal()
            #magmc = mag + np.random.normal(np.zeros(len(mag)),mag_err)
            magmc = mag + np.random.normal(np.zeros(len(mag)),tot_err)
            result_photometry = minimize(fit_photometry,mc_photoparams,args=(wave_splice_snifs,wave_splice_spex,wave_whole,flux_whole,err_whole,photsysfile,systems,bands,magmc,tot_err,ebvmc),method='Nelder-Mead',options={'maxiter':10})
            mc_photparams=result_photometry.x
            fluxmc, errmc = warp_spec(mc_photparams,wave_splice_snifs,wave_splice_spex,wave_whole,flux_whole,err_whole) # alter the spectrum accordingly
            result_spectrum = minimize(fit_spectrum,mc_modelparams,args=(wave,flux,err,snr,wave_splice_snifs,wave_splice_spex,mc_photparams,filter,gridparams,grid,feh,feh_err,ebvmc),method='Nelder-Mead',options={'maxiter':5})
            mc_params = result_spectrum.x
            teffmc.append(mc_params[0])
            loggmc.append(mc_params[1])
            metalmc.append(mc_params[2])
            #ebvmc.append(mc_params[3]) # EXPERIMENTAL

            patched_spectrum, patched_err, ratio = patch_spectrum(mc_params,wave,flux,err,snr,filter,gridparams,grid)

            wave_append = np.array(max(wave) + (wave_max - max(wave))*np.arange(nappend)/(1.*nappend))
            flux_append, err_append = append_spectrum(gridfile,teffgrid,logggrid,metalgrid,rv_model,ratio,mc_params,wave_append)
            wave_whole = np.concatenate([wave,wave_append])
            flux_whole = np.concatenate([patched_spectrum,flux_append])
            err_whole = np.concatenate([patched_err,err_append])
            indices = np.argsort(wave_whole)
            wave_whole = wave_whole[indices]
            flux_whole = flux_whole[indices]
            err_whole = err_whole[indices]
            
            fbolval = fluxcalc(wave_whole,fluxmc)
            fbolmc.append(fbolval)

        print('Results for ',starname)
        print('Reduced chi2 for spectral fitting = ',chisqmodel)
        print('Chi2 for photometry fitting = ',chisqphoto)
        print('Bolometric flux = ',fbol,'+/-',np.std(fbolmc))
        print('Teff = ',bestfit_params[0], '+/-',np.std(teffmc))
        print('log g = ',bestfit_params[1], '+/-',np.std(loggmc))
        print('[Fe/H] = ',bestfit_params[2], '+/-',np.std(metalmc))
        #print('E(B-V) = ',bestfit_params[3], '+/-',np.std(ebvmc)) # EXPERIMENTAL
        print('[Fe/H] from SpeX  = ', feh, '+/-',feh_err)
        print('Systematic error in photometry = ',sys_err)
        # add or update information to catalog file and pickle
        star = starinfo()
        star.spectrum.wave = wave_whole
        star.spectrum.s = fluxmod
        star.parameters.teff = bestfit_params[0]
        star.parameters.teff_err = np.std(teffmc)
        star.parameters.logg = bestfit_params[1]
        star.parameters.logg_err = np.std(loggmc)
        star.parameters.metal = bestfit_params[2]
        star.parameters.metal_err = np.std(metalmc)
        star.parameters.irmetal = feh
        star.parameters.irmetal_err = feh_err
        star.parameters.fbol = fbol
        star.parameters.fbol_err = np.std(fbolmc)

        observations = []
        for system, band, mval, errval in zip(systems, bands, mag, mag_err):
            observation = photometry()
            observation.system = system
            observation.bandpass = band
            observation.mag = mval
            observation.mag_err = errval
            observations.append(observation)

        star.photometry = observations
        
        filepath = Path(catalogfile)
        if filepath.is_file():
            catalog = pd.read_pickle(catalogfile)
            catalog[starname] = star
        else:
            catalog = dict()
            catalog[starname] = star

        with open(catalogfile,'wb') as output:
            pickle.dump(catalog,output,pickle.HIGHEST_PROTOCOL) 

        # make a plot
        flux_model = interpolate_spectrum(bestfit_params,gridparams,grid,wave)
        mags_syn = []
        mags_syn_err = []
        waveb = []
        for system, band, magval, magval_err in zip(systems,bands,mag,mag_err): # generate synthetic magnitudes
            transmission, zeropoint, waveband = build_bandpass(photsysfile,system,band,wave_whole)
            magsynth, magsynth_err = synth_mag(wave_whole,fluxmod,errmod,transmission,zeropoint)
            mags_syn.append(np.asscalar(magsynth))
            mags_syn_err.append(np.asscalar(magsynth_err))
            waveb.append(np.asscalar(waveband))

        # normalize flux model for plotting   DELETE BECAUSE REDUNDANT
        fluxmod, errmod = warp_spec(photparams_best,wave_splice_snifs,wave_splice_spex,wave,flux,err) # alter the spectrum accordingly
        

        r = fluxmod/flux_model 
        r[np.isnan(r)] = 0.
        which = [(r != 0.)]
        r0 = np.median(r[which])
        flux_model = r0*flux_model
        
        plot_star(starname,plotfile,wave,fluxmod,wave,flux_model,waveb,mag,mag_err,mags_syn,mags_syn_err)

main()


