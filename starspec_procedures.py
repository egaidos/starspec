def plot_star(starname,plotfile,wave,flux,wave_model,flux_model,waveb,mags,mag_errs,mags_syn,mags_syn_err):  # makes a nice plot

    import numpy as np
    import matplotlib.pyplot as plt

    plt.close('all')
    fig = plt.figure(figsize=(5,7))
    plt.rc('xtick',labelsize=6)
    plt.rc('ytick',labelsize=6)
    xmins = [0.55,1.0,1.43,1.95,1.165]
    xmaxes = [0.93,1.3,1.80,2.40,1.295]
    titles = [starname+': Visible','J','H','K','RV/resolution analysis range']
    for iplot, xmin, xmax, title in zip(np.arange(5),xmins,xmaxes,titles):
        plt.title(title,fontsize=8)
        ax = fig.add_subplot(6,1,iplot+1)
        plt.xlim([xmin,xmax])
        plt.title(title,fontsize=8)
        yvalues = flux.compress((wave > xmin) & (wave < xmax))
        plt.ylim([min(yvalues),max(yvalues)])
        plt.plot(wave,flux,'k')
        plt.plot(wave_model,flux_model,'r')
        
    dmag = np.array(mags)-np.array(mags_syn)
    dmag_err = np.sqrt(np.array(mags_syn_err)**2 + np.array(mag_errs)**2)
    ax = fig.add_subplot(6,1,6)
    plt.plot(waveb,dmag,'ko')
    plt.errorbar(waveb,dmag,yerr=dmag_err,fmt='none')
    plt.tight_layout()
    plt.savefig(plotfile)
    plt.show(block='True')

########################################################################

# interpolates spectrum on a grid

def interpolate_spectrum(parameters,gridparams,grid,wave):

# this is a regular grid interpolation which loops over wavelength (very slow)
# need to improve by a matrix.

    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    spectrum = wave*0.
    for n in np.arange(len(wave)):
        spec_interp = RegularGridInterpolator(gridparams,grid[:,:,:,n],bounds_error=False)
        #spec_interp = RegularGridInterpolator(gridparams,grid[:,:,:,n],bounds_error=True)
        spectrum[n] = spec_interp(parameters)

    return spectrum

##########################################################################

# calculate bolometric flux from spectrum

def fluxcalc(wave,flux):

    import numpy as np
    
    dwave = (wave - np.roll(wave,1))*10000  # convert back to Angstroms
    dwave = dwave*(dwave > 0.)
    fave = (flux + np.roll(flux,1))/2.
    fbol = np.sum(fave*dwave) # integrate over spectrum (simple now, should replace with something better)
    frj = flux_rj(wave,flux) # Rayleigh-Jeans tail contribution
    fbol = fbol + frj # add in
    return fbol

######################################################################

def build_grid(gridfile,nbest,res,wave,ebv,flux,err,filter,wave_rv,flux_rv,feh,feh_err):

    # build the comparison grid of models

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from astropy.convolution import convolve, Gaussian1DKernel
    from scipy.signal import savgol_filter
    from PyAstronomy.pyasl import crosscorrRV
    from PyAstronomy.pyasl import vactoair2 as vactoair
    from extinction import fm07,apply

    from astropy.io import fits

    print('Building the Grid....')

    nn = 10
    rvmax = 400
    rvmin = -400
    drv = 2.
    c = 3.0e5
    gauss_kernel = Gaussian1DKernel(nn)

    # load the grid
    f = fits.open('../Grid/'+gridfile)
    g = f[1].data
    #header = g['HEADER'][0]
    modelspec = g['SPECTRUM'][0]
    teff = np.array(g['TEFF'][0])
    logg = np.array(g['LOGG'][0])
    metal = np.array(g['METAL'][0])
    a_fe = np.array(g['A_FE'][0])

    #junk = header[8]
    #junk = np.array(junk.split())
    #nlambda = int(np.asscalar((junk[2])))
    #junk = header[9]
    #junk = np.array(junk.split())
    #lambda0 = float(np.asscalar(junk[1]))
    #junk = header[10]
    #junk = np.array(junk.split())
    #dlambda = float(np.asscalar(junk[1]))
    #modelwave = lambda0 + dlambda*np.arange(nlambda) # for original PHOENIX
    modelwave = np.array(g['WAVE'][0]) # for Goettingen PHOENIX
    nwave,nmodels = modelspec.shape
    nwave = int(nwave)
    nmodels = int(nmodels)
 
    model_vac = np.array(modelwave) # convert wavelengths from vacuum to air and to microns
    model_vac2 = model_vac.compress(model_vac > 2000.)
    model_wave = vactoair(model_vac2,mode='edlen53')/10000. # convert to air
    npts = nn*res*np.log(np.max(model_wave)/np.min(model_wave))
    wave_log = np.min(model_wave)*np.exp(np.arange(npts)/(1.*npts)*np.log(np.max(model_wave)/np.min(model_wave)))

    av = ebv*3.1
    #extinct_all = extinction.fm07(wave*10000.,av)/av_ref   # approximate extinction per unit magnitude
    #extinct = extinct_all.compress(filter==1)

    #model_compare = []
    rvbest = []
    chisq = []
    #avmodels = []
    #av_max = 4.
    #av_values = av_max*np.arange(100)/100.


    for imodel in np.arange(nmodels):

        model_spec= np.array(modelspec[:,imodel])*1.0e-8 # convert from per cm to per A
        model_spec = model_spec.compress(model_vac > 2000.)
        #model_spec = model_spec*10**(-0.4*extinction.fm07(model_vac,av)) # apply extinction  (EXPERIMENTAL)

        # interpolation function
        model_interp = interp1d(model_wave,model_spec,kind='linear',bounds_error=False,fill_value=0.)
        model_prelim = model_interp(wave)
        model_prelim = apply(fm07(wave*10000,ebv,3.1),model_prelim)
        fr = flux.compress(filter == 1.)/model_prelim.compress(filter == 1)
        
        # here we need to determine best-fit extinction
        #x = []
        
        #for av_val in av_values:
        #    x1 = np.sum(fr*10**(-0.4*extinct*av_val))/np.sum(10**(-0.8*extinct*av_val))
        #    x2 = np.sum(extinct*fr*10**(-0.4*extinct*av_val))/np.sum(extinct*10**(-0.8*extinct*av_val))
        #    x.append(abs(x1/x2-1.))
        #av_best = av_values[np.argmin(x)]
        #avmodels.append(av_best)
        
        #ratval = np.sum(fr*10**(-0.4*extinct*av_best))/np.sum(10**(-0.8*extinct*av_best))*10**(-0.4*extinct*av_best)
        ratval = np.median(fr)
        model_prelim = model_prelim.compress(filter==1)*ratval
        sig = (flux.compress(filter==1)-model_prelim)/err.compress(filter==1)
        #plt.plot(wave.compress(filter==1),err.compress(filter==1))
        #plt.show(block=True)
        chisqval = np.sum(sig**2)*(1 + ((feh-metal[imodel])/feh_err)**2) # disabled for now 
        chisq.append(chisqval)
        print('Model params =',teff[imodel],logg[imodel],metal[imodel],av_best,chisqval)
    
    indices = np.argsort(chisq)
    bestmodels = indices[0:nbest]
    chisq = np.array(chisq)
 
    teffbest = np.sort(np.unique(teff[bestmodels]))
    loggbest = np.sort(np.unique(logg[bestmodels]))
    metalbest = np.sort(np.unique(metal[bestmodels]))
    #avmodels = np.array(avmodels)
    #avbest = np.sort(np.unique(avmodels[bestmodels]))
    #print('Best Av = ',avmodels[indices[0]])

    bestparams = [teff[indices[0]],logg[indices[0]],metal[indices[0]]]
    print('Best parameters = ',bestparams)
    plt.pause(5)
    teffmin = np.min(teffbest)
    teffmax = np.max(teffbest)
    loggmin = np.min(loggbest)
    loggmax = np.max(loggbest)
    metalmin = np.min(metalbest)
    metalmax = np.max(metalbest)
    if teffmin == teffmax:
        teffmin = teffmin - 100.
        teffmax = teffmax + 100.
    if loggmin == loggmax:
        loggmin = loggmin - 0.5
        loggmax = loggmax + 0.5
    if metalmin == metalmax:
        metalmin = metalmin - 0.5
        metalmax = metalmax + 0.5

    gridmodels = np.arange(nmodels).compress((teff >= teffmin) & (teff <= teffmax) & (logg >= loggmin) & (logg <= loggmax) & (metal >= metalmin) & (metal <= metalmax))

    grid = np.zeros((len(teffbest),len(loggbest),len(metalbest),len(wave)))

    print('Creating subgrid with ',len(gridmodels),' models')
    for imodel in gridmodels:
        model_spec = np.array(modelspec[:,imodel])*1.0e-8 # convert from per cm to per A
        model_spec = model_spec.compress(model_vac > 2000.)
        
        # interpolate model spectrum to logarithmic wavelengths
        model_interp = interp1d(model_wave,model_spec,kind='linear',bounds_error=False,fill_value=0.)
        model_log = model_interp(wave_log)
        model_conv = convolve(model_log,gauss_kernel) # convolve model with Gaussian
        smooth_model = savgol_filter(model_conv,501,2)
        model_norm = model_conv/smooth_model
        wavelim = wave_log.compress((wave_log < 1.01*np.max(wave_rv)) & (wave_log > 0.99*np.min(wave_rv)))
        modellim = model_norm.compress((wave_log < 1.01*np.max(wave_rv)) & (wave_log > 0.99*np.min(wave_rv)))
        
        # find Doppler shift of model
        rv, cc = crosscorrRV(wave_rv,flux_rv,wavelim,modellim,rvmin,rvmax,drv,mode='doppler')
        index = np.argmax(cc)
        
        rvbest.append(rv[index])
        wave_shift = wave_log*(1 + rv[index]/c)
            
        model_interp = interp1d(wave_shift,model_conv,kind='linear',bounds_error=False,fill_value=0.)

        i = np.arange(len(teffbest)).compress(teff[imodel] == teffbest)
        j = np.arange(len(loggbest)).compress(logg[imodel] == loggbest)
        k = np.arange(len(metalbest)).compress(metal[imodel] == metalbest)
        #gridflux = model_interp(wave)*10**(-0.4*extinct_all*np.asscalar(avmodels[imodel])) # apply extinction at this step for consistency with the chi-squared minimization
        gridflux = model_interp(wave)
        grid[i,j,k,:] = gridflux

    return teffbest, loggbest, metalbest, grid, rvbest, bestparams

###################################################################

def build_spectrum(specfile,starname,nstar): # build spectrum of star from SNIFS and SpeX spectra

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.interpolate import interp1d

    extension = specfile.rpartition(".")[-1]

    if extension == "h5":
        f = pd.read_hdf('../Data/'+specfile,'table')
        wave_blue = np.array(f['SNIFSB_WAVE'][nstar])/10000.
        flux_blue = np.array(f['SNIFSB_FLUX'][nstar])
        err_blue = np.array(f['SNIFSB_ERROR'][nstar])
        if (np.median(err_blue) > np.median(flux_blue)):  # checks for someone taking the square root of the error
            err_blue = err_blue**2 
        wave_red = np.array(f['SNIFSR_WAVE'][nstar])/10000.
        flux_red = np.array(f['SNIFSR_FLUX'][nstar])
        err_red = np.sqrt(np.array(f['SNIFSR_ERROR'][nstar]))
        if (np.median(err_red) > np.median(flux_red)):
            err_red = err_red**2 
        wave_spex = np.array(f['SPEX_WAVE'][nstar])
        flux_spex = np.array(f['SPEX_FLUX'][nstar])
        err_spex = np.array(f['SPEX_ERROR'][nstar])
        if (np.median(err_spex) > np.median(flux_spex)):
            err_spex = err_spex**2 
        
    if extension == "pkl":
        
        f = pd.read_pickle('../Data/'+specfile)
        wave_blue = np.array(f[starname].snifs.blue.wave)
        flux_blue = np.array(f[starname].snifs.blue.flux)
        err_blue = np.array(f[starname].snifs.blue.err)
        wave_red = np.array(f[starname].snifs.red.wave)
        flux_red = np.array(f[starname].snifs.red.flux)
        err_red = np.array(f[starname].snifs.red.err)
        wave_spex = np.array(f[starname].spex.wave)
        flux_spex = np.array(f[starname].spex.flux)
        err_spex = np.array(f[starname].spex.err)

    goodspex = np.logical_not(np.isnan(flux_spex) & np.isnan(err_spex))
    goodblue = np.logical_not(np.isnan(flux_blue) & np.isnan(err_blue))
    goodred = np.logical_not(np.isnan(flux_red)  & np.isnan(err_red))

    flux_spex = flux_spex[goodspex]
    err_spex = err_spex[goodspex]
    wave_spex = wave_spex[goodspex]

    flux_blue = flux_blue[goodblue]
    err_blue = err_blue[goodblue]
    wave_blue = wave_blue[goodblue]

    flux_red = flux_red[goodred]
    err_red = err_red[goodred]
    wave_red = wave_red[goodred]

    #    flux_spex[np.isnan(flux_spex)] = 0.
    #    err_spex[np.isnan(err_spex)] =  1.0e-30 #99.0
    #    flux_blue[np.isnan(flux_blue)] = 0.
    #    err_blue[np.isnan(err_blue)] = 1.0e-30 #99.0
    #    flux_red[np.isnan(flux_red)] = 0.
    #    err_red[np.isnan(err_red)] = 1.0e-30 #99.0

    if (np.isnan(np.sum(flux_blue)) | np.isnan(np.sum(flux_red)) | np.isnan(np.sum(flux_spex))):
#    if (np.isnan(np.sum(f['SNIFSB_FLUX'][nstar])) | np.isnan(np.sum(f['SNIFSR_FLUX'][nstar])) | np.isnan(np.sum(f['SPEX_FLUX'][nstar]))):

        wave = None
        flux = None
        err = None
        wave_splice_snifs = 0.
        wave_splice_spex = 0.
    else:
#        wave_blue = np.array(f['SNIFSB_WAVE'][nstar])/10000.
#        flux_blue = np.array(f['SNIFSB_FLUX'][nstar])
#        err_blue = np.array(f['SNIFSB_ERROR'][nstar])
#        wave_red = np.array(f['SNIFSR_WAVE'][nstar])/10000.
#        flux_red = np.array(f['SNIFSR_FLUX'][nstar]) 
#        err_red = np.array(f['SNIFSR_ERROR'][nstar])
#        wave_spex = np.array(f['SPEX_WAVE'][nstar])
#        flux_spex = np.array(f['SPEX_FLUX'][nstar])
#        err_spex = np.array(f['SPEX_ERROR'][nstar])

        snr_spex = flux_spex/err_spex
    
#        flux_spex[np.isnan(flux_spex)] = 0.
#        err_spex[np.isnan(err_spex)] = 1.0e-30 # 99.0
#        flux_blue[np.isnan(flux_blue)] = 0.
#        err_blue[np.isnan(err_blue)] = 1.0e-30 #99.0
#        flux_red[np.isnan(flux_red)] = 0.
#        err_red[np.isnan(err_red)] = 1.0e-30 #99.0
    
        wave_spex_all = wave_spex[:]
        flux_spex_all = flux_spex[:]
        err_spex_all = err_spex[:]

        # determine overlap and ratio of blue to red spectra
        wave_red_over_blue = wave_red.compress(wave_red < np.max(wave_blue))
        flux_red_over_blue = flux_red.compress(wave_red < np.max(wave_blue))
        interp_overlap = interp1d(wave_blue,flux_blue,kind='linear')
        flux_blue_over_red = interp_overlap(wave_red_over_blue)
        ratio = np.median(flux_red_over_blue/flux_blue_over_red)
    
        # adjust blue spectrum
        flux_blue = ratio*flux_blue
        err_blue = ratio*err_blue

        # figure optimal splice point
        snr_blue = flux_blue/err_blue
        snr_red = flux_red/err_red

        interp_overlap = interp1d(wave_blue,snr_blue,kind='linear')
        snr_blue_over_red = interp_overlap(wave_red_over_blue)
        snr_red_over_blue = snr_red.compress(wave_red <= np.max(wave_red_over_blue))
        exceed = np.sum(1.*(snr_red_over_blue > snr_blue_over_red))
        if (exceed == 0):
            wave_splice_snifs = np.max(wave_red_over_blue)
        else:
            wave_splice_snifs = np.min(wave_red_over_blue[snr_red_over_blue > snr_blue_over_red])

        # trim blue and red spectra
        flux_blue = flux_blue.compress(wave_blue < wave_splice_snifs)
        err_blue = err_blue.compress(wave_blue < wave_splice_snifs)
        wave_blue = wave_blue.compress(wave_blue < wave_splice_snifs)
        flux_red = flux_red.compress(wave_red > wave_splice_snifs)
        err_red = err_red.compress(wave_red > wave_splice_snifs)
        wave_red = wave_red.compress(wave_red > wave_splice_snifs)
    
        # concatenate blue and red spectra
        wave = np.concatenate([wave_blue,wave_red])
        flux = np.concatenate([flux_blue,flux_red])
        err = np.concatenate([err_blue,err_red])

        # find overlap and calculate ratio spex/snifs (if not then just concatenate)
        if (np.max(wave) > np.min(wave_spex)): # check if overlapd
            wave_snifs_over_spex = wave.compress(wave > np.min(wave_spex))
            flux_snifs_over_spex = flux.compress(wave > np.min(wave_spex))
            interp_overlap = interp1d(wave_spex,flux_spex,kind='linear')
            wave_spex_over_snifs = wave_spex.compress(wave_spex < np.max(wave))
            flux_spex_over_snifs = interp_overlap(wave_snifs_over_spex)
            ratio = np.median(flux_snifs_over_spex/flux_spex_over_snifs)
            # correct spex
            flux_spex = ratio*flux_spex
            err_spex = ratio*err_spex

            # figure optimal splice point
            snr_spex = flux_spex/err_spex
            snr = flux/err
            interp_overlap = interp1d(wave,snr,kind='linear')
            snr_snifs_over_spex = interp_overlap(wave_spex_over_snifs)
      
            snr_spex_over_snifs = snr_spex.compress(wave_spex <= np.max(wave_spex_over_snifs))
            exceed = np.sum(1.*(snr_spex_over_snifs > snr_snifs_over_spex))
            if (exceed == 0):
                wave_splice_spex = np.max(wave_spex_over_snifs)
            else:
                wave_splice_spex = np.min(wave_spex_over_snifs[snr_spex_over_snifs > snr_snifs_over_spex])
            # trim spex
            flux_spex = flux_spex.compress(wave_spex > np.max(wave))
            err_spex = err_spex.compress(wave_spex > np.max(wave))
            wave_spex = wave_spex.compress(wave_spex > np.max(wave))
        else:
            wave_splice_spex = np.min(wave_spex)
        

        # construct full spectrum
        wave = np.concatenate([wave,wave_spex])
        flux = np.concatenate([flux,flux_spex])
        err = np.concatenate([err,err_spex])

        flux[np.isnan(err)] = 0.
        err[np.isnan(err)] = 0.

    return wave, flux, err, wave_splice_snifs, wave_splice_spex

#=====================================================================

def build_filter(wave,filterfile): # builds spectral filter file

    import numpy as np
    from astropy.io import ascii

    f = ascii.read(filterfile)
    filter_wave_low = np.array(f['col1'])/10000.
    filter_wave_high = np.array(f['col2'])/10000.

    filter = wave*0 + 1.

    for wave_low, wave_high in zip(filter_wave_low, filter_wave_high):
        filter = filter*((wave < wave_low) | (wave > wave_high))

    return filter

####################################################

def resolution(wave,flux,err,waverange,filter):    # determines spectral resolution 

    from PyAstronomy.pyasl import crosscorrRV
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.signal import savgol_filter

    # gaussian for fitting CC to find width
    def gauss(x,a,b,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+b


    wave_temp_low = 0.97 # minimum wavelength in microns
    rvmin = -300. # RV range
    rvmax = 300 
    drv = 2. # increment of RV
    c = 3.0e5 # speed of light in km/s
    minsr = 100 # minimum SNR
    nsmooth = 51
    resmin = 2000.  # minimum expected resolution

    max_wave = np.max(waverange)*(1 + 2*rvmax/c)
    min_wave = np.min(waverange)*(1 + 2*rvmin/c)
    which = ((wave > min_wave) & (wave < max_wave))
    wave_template = wave[which]
    flux_template = flux[which]
    err_template = err[which]
    smooth_template = savgol_filter(flux_template,nsmooth,2)
    flux_template = flux_template/smooth_template
    err_template = err_template/smooth_template 
    
    which = ((wave > np.min(waverange)) & (wave < np.max(waverange)))
    wave_target = wave[which]
    flux_target = flux[which]
    err_target = err[which]
    smooth_target = savgol_filter(flux_target,nsmooth,2)
    flux_target = flux_target/smooth_target
    err_target = err_target/smooth_target

    rv, cc = crosscorrRV(wave_target,flux_target,wave_template,flux_template,rvmin,rvmax,drv, mode='doppler')
    mean = sum(rv*cc)/sum(cc)
    sigma = np.sqrt(sum(cc*(rv-mean)**2)/sum(cc))
    base = np.min(cc)
    amp = np.max(cc)-base

    popt, pcov = curve_fit(gauss,rv,cc,p0=[amp,base,mean,sigma])
    sigma = popt[3]
    res = c/popt[3]
    print('Inferred SpeX resolution =',res)

    #import matplotlib.pyplot as plt
    #plt.plot(rv,cc,'k')
    #plt.plot(rv,gauss(rv,popt[0],popt[1],popt[2],popt[3]),'r')
    #plt.show(block='True')

    # return resolution and piece of spectrum to use to compare with models
    if (res < resmin):
        print('Setting resolution to minimum =',resmin)
        res = resmin
    res = 2*res # this is fudge-factor based on eye
    return res, wave_target, flux_target

###################################################

def fit_spectrum(parameters,wave,flux,err,snr,wave0,wave1,photparams,filter,gridparams,grid,feh,feh_err,ebv):  # function for spectral fit to minimize

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    from extinction import fm07, apply

    minsnr = 50
    av = 3.1*ebv

    fluxmod, errmod = warp_spec(photparams,wave0,wave1,wave,flux,err)
    flux_model = interpolate_spectrum(parameters,gridparams,grid,wave)
    #flux_model = interpolate_spectrum(parameters[0:3],gridparams,grid,wave)  # EXPERIMENTAL
    #flux_model = flux_model*10**(-0.4*extinction.fm07(wave*10000.,parameters[3])) # EXPERIMENTAL
    flux_model = apply(fm07(wave*10000,av),flux_model) # include reddening
    filter_all = filter*(1.*(snr > minsnr))
    
    r = fluxmod/flux_model
    r[np.isnan(r)] = 0.

    # Overall adjustment
    which = ((r != 0.) & (filter == 1.) & (np.isfinite(r)) & (np.isnan(r) == False))
    r0 = np.median(r[which])
    print('Model params = ',parameters)
    print('Phot params = ',photparams)
    flux_model = r0*flux_model
    
    var = (filter_all*(fluxmod-flux_model)/errmod)**2
    var[(1*np.isnan(var)) == 1.] = 0.
    chisquared = np.sum(var)/np.sum([1.*(var > 0.)])*(1 + ((feh-parameters[2])/feh_err)**2)

    # plotting routines
    plt.close('all')
    plt.rc('xtick',labelsize=6)
    plt.rc('ytick',labelsize=6)
    fig = plt.figure(figsize=(4,5))
    xmins = [0.55,1.0,1.43,1.95,1.165]
    xmaxes = [0.93,1.3,1.80,2.40,1.295]
    titles = ['Visible','J','H','K','RV/resolution analysis range']
    for iplot, xmin, xmax, title in zip(np.arange(5),xmins,xmaxes,titles):
        ax = fig.add_subplot(5,1,iplot+1)
        plt.xlim([xmin,xmax])
        plt.title(title,fontsize=8)
        yvalues = fluxmod.compress((wave > xmin) & (wave < xmax))
        plt.ylim([min(yvalues),max(yvalues)])
        plt.plot(wave,filter_all*fluxmod,'k.',markersize=0.7)
        plt.plot(wave,flux_model,'r')
    plt.tight_layout()
    plt.pause(0.1)
    plt.clf()
    if np.isnan(chisquared):
        chisquared = 1.0e9

    return chisquared
    
#########################################################
    
def patch_spectrum(parameters,wave,flux,err,snr,filter,gridparams,grid): # patch spectrum with best-fit model

    import numpy as np
    import matplotlib.pyplot as plt
    
    print('Patching spectrum....')

    minsnr = 50.
    snr_patch = 20.

    bestfit_spectrum = interpolate_spectrum(parameters,gridparams,grid,wave)

    patched_spectrum = np.array(flux[:])
    patched_err = np.array(err[:])
        
    goodparts = ((snr > minsnr) & (filter == 1.))
    model_good = bestfit_spectrum[goodparts]
    flux_good = flux[goodparts]
    
    r = flux_good/model_good

    r[np.isnan(r)] = 0.
    #plt.close()
    #plt.plot(wave[goodparts],r)
    #plt.show(block=True)
    rmed = np.nanmedian(r)
    bestfit_spectrum = bestfit_spectrum*rmed
    the_patch = (((filter == 0.) | (snr < snr_patch)) & (bestfit_spectrum > 0.))
    patched_spectrum[the_patch] = bestfit_spectrum[the_patch]
    err_patch = np.median(err[the_patch]) # use median in the patch; should do something better
    patched_err[the_patch] = err_patch 

    return patched_spectrum, patched_err, rmed

def build_bandpass(photsysfile,sys_select,band_select,wavelength):

    import numpy as np
    from scipy.io import readsav
    from scipy.interpolate import interp1d

    phot_sys = readsav(photsysfile)
    phot_systems = phot_sys.phot_systems
    wave = np.array(phot_systems.LAMBDA)
    trans = np.array(phot_systems.trans)
    bands = np.array(phot_systems.band)
    systems = np.array(phot_systems.system)
    zeropt = np.array(phot_systems.zp)
    zp_src = np.array(phot_systems.zp_source)

    i = 0
    for band in bands:
        bands[i] = str(band,'UTF-8')
        i += 1
    i = 0
    for system in systems:
        systems[i] = str(system,'UTF-8')
        i += 1

    wavefilter = wave.compress((systems == sys_select) & (bands == band_select))
    transfilter = trans.compress((systems == sys_select) & (bands == band_select))
    zeropoint = np.asscalar(zeropt.compress((systems == sys_select) & (bands == band_select)))

    wavefilter = wavefilter[0]
    transfilter = transfilter[0]
    trans_interp = interp1d(wavefilter,transfilter,kind='linear',bounds_error=False, fill_value=0.)
    transmission = trans_interp(wavelength)
    transmission = transmission*(transmission > 0.)
    waveband = np.sum(transmission*wavelength)/np.sum(transmission)

    return transmission, zeropoint, waveband

########################################################

def synth_mag(wave,flux,err,transmission,zeropoint):  # synthesize photometry from spectra

    import numpy as np
    import matplotlib.pyplot as plt

    dwave = wave-np.roll(wave,1)
    dwave = dwave*(dwave > 0.)
    f = flux*transmission
    fave = (f + np.roll(f,1))/2. # trapezoidal integration
    tave = (transmission + np.roll(transmission,1))/2.
    var = err**2*transmission
    vave = (var + np.roll(var,1))/2.
    flux_tot = np.sum(fave*dwave)/np.sum(tave*dwave)
    err_tot = np.sqrt(np.sum(vave*dwave)/np.sum(tave*dwave))
    if (err_tot > 1e-11):  #if unreasonable high noise then spectrum has sigma in place of variance
        err_tot = np.sum(vave*dwave)/np.sum(tave*dwave)
    mag = -2.5*np.log10(flux_tot/zeropoint)
    mag_err = err_tot/flux_tot*2.5/np.log(10.)

    return mag, mag_err

##############################################################

# append spectrum with model at longer wavelengths

def append_spectrum(gridfile,teffgrid,logggrid,metalgrid,rv_model,ratio,bestparams,wave_append):

    # wavelengths are shifted but spectrum is not convolved (this is only for photometry)

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import interp1d
    from astropy.convolution import convolve, Gaussian1DKernel
    from scipy.signal import savgol_filter
    from PyAstronomy.pyasl import crosscorrRV
    from PyAstronomy.pyasl import vactoair2 as vactoair
    from astropy.io import fits

    print('Appending spectrum....')

#    from interpolate_spectrum import interpolate_spectrum

    c = 3.0e5

    # load the grid
    f = fits.open('../Grid/'+gridfile)
    g = f[1].data
    #header = g['HEADER'][0]
    modelspec = g['SPECTRUM'][0]
    teff = np.array(g['TEFF'][0])
    logg = np.array(g['LOGG'][0])
    metal = np.array(g['METAL'][0])
    #junk = header[8]
    #junk = np.array(junk.split())
    #nlambda = int(np.asscalar((junk[2])))
    #junk = header[9]
    #junk = np.array(junk.split())
    #lambda0 = float(np.asscalar(junk[1]))
    #junk = header[10]
    #junk = np.array(junk.split())
    #dlambda = float(np.asscalar(junk[1]))
    #modelwave = lambda0 + dlambda*np.arange(nlambda)
    modelwave = np.array(g['WAVE'][0])
    nwave,nmodels = modelspec.shape
    nwave = int(nwave)
    nmodels = int(nmodels)
 
    model_vac = np.array(modelwave) # convert wavelengths from vacuum to air and to microns
    model_vac2 = model_vac.compress(model_vac > 2000.) # cuts off at 2000A
    model_wave = vactoair(model_vac2,mode='edlen53')/10000.

    gridmodels = np.arange(nmodels).compress((teff >= np.min(teffgrid)) & (teff <= np.max(teffgrid)) & (logg >= np.min(logggrid)) & (logg <= np.max(logggrid)) & (metal >= np.min(metalgrid)) & (metal <= np.max(metalgrid)))
    
    grid = np.zeros((len(teffgrid),len(logggrid),len(metalgrid),len(wave_append)))
    for imodel, rv in zip(gridmodels,rv_model):

        model_spec = np.array(modelspec[:,imodel])*1.0e-8 # convert from per cm to per A
        model_spec = model_spec.compress(model_vac > 2000.) # cuts off at 2000A
        wave_shift = model_wave*(1 + rv/c)
   
        model_interp = interp1d(wave_shift,model_spec,kind='linear',bounds_error=False,fill_value=0.)         

        i = np.arange(len(teffgrid)).compress(teff[imodel] == teffgrid)
        j = np.arange(len(logggrid)).compress(logg[imodel] == logggrid)
        k = np.arange(len(metalgrid)).compress(metal[imodel] == metalgrid)
        grid[i,j,k,:] = model_interp(wave_append)

    flux_append = np.array(ratio*interpolate_spectrum(bestparams,[teffgrid,logggrid,metalgrid],grid,wave_append))
    err_append = flux_append[:]*0  # temporary: should change based on error in best-fit model parameters

    return flux_append, err_append

######################################################

def get_photometry(photofile,hashfile,starname,exclude):  # retrieve photometry from database

    import pickle
    import pandas as pd
    from astropy.io import ascii
    import numpy as np
    import re
    
    photohash = ascii.read(hashfile) # file that maps to bandpass names to standard nomenclature
    filters = np.array(photohash['FILTER'])
    systems = np.array(photohash['SYSTEM'])
    bands = np.array(photohash['BAND'])    
    saturation = np.array(photohash['SATURATION']) # saturation magnitude
    switch = np.array(photohash['SWITCH']) # selects which photometry to use

    df = pd.read_pickle(photofile)
    starnames = np.array(df.index)
    indices = np.arange(len(starnames))
    nstar = np.asscalar(indices.compress(starnames == starname))

    # customizing of filters to exclude
    if exclude != None:
        exclude_filters = exclude.split("-")
        for exclude_filter in exclude_filters:
            switch[(bands == exclude_filter)] = 0.
    
    systems_out = []
    bands_out = []
    mags_out = []
    mag_err_out = []
    columns = np.array(df.columns)
    regex = re.compile('err')
    observations = []
    for col in columns:
        for m in [regex.search(col)]:
            if not m:
                observations.append(col)
    indices = np.arange(len(filters))
    for observation in observations:
        if observation in filters:
            mag = df[observation][nstar]
            mag_err= df[observation+'_err'][nstar]
            index = indices[(filters == observation)]
            if not(np.isnan(mag) | (mag < saturation[index]) | (mag < 0) | (switch[index] == 0)):
                systems_out.append(systems[index])
                bands_out.append(bands[index])
                mags_out.append(mag)
                mag_err_out.append(mag_err)

    return systems_out, bands_out, mags_out, mag_err_out

################################################

def fit_photometry(param,wave0,wave1,wave,flux,err,photsysfile,systems,bands,mags,mag_errs,ebv): # fit photometry with synthetic photometry

    import matplotlib.pyplot as plt
    import numpy as np
    from extinction import fm07,apply

    #av = 3.1*ebv
    #fluxred = apply(fm07(wave*10000,av),flux)
    #err_red = fluxred/flux*err
    fluxmod, errmod = warp_spec(param,wave0,wave1,wave,flux,err)
    
    chi2 = 0.
    mags_syn = []
    mags_syn_err = []
    waveb = []
    for system, band, mag, mag_err in zip(systems,bands,mags,mag_errs): # generate synthetic magnitudes
        transmission, zeropoint, waveband = build_bandpass(photsysfile,system,band,wave)
        magsynth, magsynth_err = synth_mag(wave,fluxmod,errmod,transmission,zeropoint)
        mags_syn.append(np.asscalar(magsynth))
        mags_syn_err.append(np.asscalar(magsynth_err))
        waveb.append(np.asscalar(waveband))
        chi2 = chi2 + ((mag - magsynth)**2/(mag_err**2 + magsynth_err**2))

    dmag = np.array(mags)-np.array(mags_syn)
    dmag_err = np.sqrt(np.array(mags_syn_err)**2 + np.array(mag_errs)**2)
    plt.plot(waveb,dmag,'ko')
    plt.ylim([-1,1])
    plt.errorbar(waveb,dmag,yerr=dmag_err,fmt='none')
    for x,y,lab in zip(waveb,dmag,bands):
        plt.text(x+0.03,y,lab[0])
    plt.pause(0.2)
    plt.clf()
    return chi2

################################################

def warp_spec(param,wave0,wave1,wave,flux,err):

    # modifies flux according to four-parameter model
    # param[0] = overall multiplicative factor
    # param[1] = offset of spectrum in SNIFS blue channel relative to the red channel
    # param[2] = offset of spectrum in Spex relative to SNIFS red channel
    # param[3] = power-law wavelength dependence along SpeX (to account for slit effects)

    import numpy as np
    wave2 = 2.56 # bondary between SpeX and longer data (e.g. WISE)
    
    fluxmod = np.asscalar(param[0])*flux[:]
    errmod = np.asscalar(param[0])*err[:]

    blue = (wave < wave0)
    fluxmod[blue] = np.asscalar(param[1])*fluxmod[blue]
    errmod[blue] = np.asscalar(param[1])*errmod[blue]

    jhk = ((wave > wave1) & (wave < wave2))
    fluxmod[jhk] = np.asscalar(param[2])*(wave[jhk]/wave1)**np.asscalar(param[3]-1.)*fluxmod[jhk]
    errmod[jhk] = np.asscalar(param[2])*(wave[jhk]/wave1)**np.asscalar(param[3]-1.)*errmod[jhk]

    wise = (wave > wave2)
    fluxmod[wise] = np.asscalar(param[2])*(wave2/wave1)**np.asscalar(param[3]-1.)*fluxmod[wise]
    errmod[wise] = np.asscalar(param[2])*(wave2/wave1)**np.asscalar(param[3]-1.)*errmod[wise]

    # alternative model for SpeX slit effect
    #fluxmod[(wave > wave1) & (wave < wave2)] = np.asscalar(param[2])*(1 + np.asscalar(param[3])*(wave[(wave > wave1) & (wave < wave2)]/wave1))*fluxmod[(wave > wave1) & (wave < wave2)]
    #errmod[(wave > wave1) & (wave < wave2)] = np.asscalar(param[2])*(1 + np.asscalar(param[3])*(wave[(wave > wave1) & (wave < wave2)]/wave1))*errmod[(wave > wave1) & (wave < wave2)]
    #fluxmod[(wave > wave2)] = np.asscalar(param[2])*(1 + np.asscalar(param[3])*(wave2/wave1))*fluxmod[(wave > wave2)]
    #errmod[(wave > wave2)] = np.asscalar(param[2])*(1 + np.asscalar(param[3])*(wave2/wave1))*errmod[(wave > wave2)]


    return fluxmod, errmod

#####################################################

def flux_rj(wave,flux):

    import numpy as np

    # calculates flux contribution from the Rayleigh-Jeans extension of the spectrum

    nfit = 21 # number of points at end to fit
    wavemax = max(wave)
    n = len(flux)
    frj = np.median(flux[n-nfit-2:n-1]*(wave[n-nfit-2:n-1]/max(wave))**4)*max(wave)/3.*10000.
    
    return frj

#######################################################

def equiv_width(wavevac,flux,flux_err,wavec,window,blue1,blue2,red1,red2): # calculates equivalent width

    # note input wavelength is in vacuum

    import numpy as np
    import matplotlib.pyplot as plt

    wavemax = wavec + window/2.
    wavemin = wavec - window/2.
    
    # compute psuedo continuum
    
    which = (((wavevac > blue1) & (wavevac < blue2)) | ((wavevac > red1) & (wavevac < red2)))
    pseudocont = flux[which]
    pseudowt  = 1./flux_err[which]**2
    lambdacont = wavevac[which]

    par = np.polyfit(lambdacont,pseudocont,1,w=pseudowt)
    fcont = par[1] + par[0]*wavevac

    which = ((wavevac > wavemin) & (wavevac < wavemax))
    dwave = np.abs(np.roll(wavevac,1) - np.roll(wavevac,-1))/2.

    # returns equivalent width in Angstroms
    ew = np.sum((1. - flux[which]/fcont[which])*dwave[which])*10000.0
#    plt.clf()
#    plt.plot(wavevac[which],flux[which],'ko')
#    plt.plot(wavevac[which],fcont[which],'r')
#    plt.show(block=True)
    return ew

###############################################

def irwater(wavevac,flux):

    import numpy as np

    water_j = np.median(flux[((wavevac > 1.210) & (wavevac < 1.230))])*np.median(flux[((wavevac > 1.331) & (wavevac < 1.351))])/np.median(flux[((wavevac > 1.313) & (wavevac < 1.333))])**2
    water_h = np.median(flux[((wavevac > 1.595) & (wavevac < 1.615))])*np.median(flux[((wavevac > 1.760) & (wavevac < 1.780))])/np.median(flux[((wavevac > 1.680) & (wavevac < 1.700))])**2
    water_k = np.median(flux[((wavevac > 2.070) & (wavevac < 2.090))])*np.median(flux[((wavevac > 2.360) & (wavevac < 2.380))])/np.median(flux[((wavevac > 2.235) & (wavevac < 2.255))])**2

    return (water_j, water_h, water_k)

#############################################
                        
def metallicity(wave,flux,flux_err,rv_nom):
              
    c = 300000.
        
    import numpy as np
    from astropy.io import ascii
    from PyAstronomy.pyasl import airtovac

    wavevac = airtovac(wave*10000.)/10000. # convert to vacuum
    waverest = wavevac*(1 - rv_nom/c) # convert to rest frame

    features = ascii.read('ir_features.txt')
    f = np.array(features['F'])
    wavec = np.array(features['Center'])
    width = np.array(features['Width'])
    blue1 = np.array(features['Blue1'])
    blue2 = np.array(features['Blue2'])
    red1 =  np.array(features['Red1'])
    red2 =  np.array(features['Red2'])
    

    # compute equivalent widths
    ew = np.zeros(23)
    for fval,wc,wd,b1,b2,r1,r2 in zip(f,wavec,width,blue1,blue2,red1,red2):
        ew[fval] = equiv_width(waverest,flux,flux_err,wc,wd,b1,b2,r1,r2)

    water_j, water_h, water_k = irwater(waverest,flux)

    feh_j = 0.2860*ew[10] + 0.1929*ew[9] + 0.2161*ew[12] - 0.2667*ew[13] + 0.2613*water_j - 1.3406
    feh_h = 0.4565*ew[17] + 0.410*ew[14] - 0.1845*ew[18] - 2.1062*water_h + 1.2844
    feh_k = 0.1963*ew[19] + 0.0681*ew[22] + 0.0134*ew[20] + 0.7839*water_k - 1.9193
    
    feh = np.sum(feh_j/0.12**2 + feh_h/0.09**2 + feh_k/0.08**2)/(1./0.12**2 + 1/0.09**2 + 1/0.08**2)
    #feh_err = np.sqrt(1./(1/0.12**2 + 1/0.09**2 + 1/0.08**2))
    feh_err = np.std([feh_j,feh_h,feh_k]) # use standard devation between estimates

    return (feh, feh_err, feh_j, feh_h, feh_k)



