from IPython.display import Image

import warnings
warnings.filterwarnings('ignore')

import eleanor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.modeling import models
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import ascii
from astropy import table
from astropy.io.votable import parse
from lightkurve.correctors import PLDCorrector
from astropy.io import fits
import lightkurve as lk
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default
import time
import multiprocessing as mp
import batman
from astropy.timeseries import BoxLeastSquares
from pylab import *
from IPython.display import Image,display
import math
from astroquery.mast import Observations
from astropy.convolution import convolve, Box1DKernel
import os
from tglc.quick_lc import tglc_lc
import glob

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


plt.rcParams['figure.dpi'] = 120

binsizedays = 30/60/24

##############################################
# New and updated version for testing
# Will hopefully match the outputs from the test script above after integration of new analysis
#############################################


def injectplanetbeforestitch(arg):
    idx,row = arg
    #normalized_corrected = lcbin
    depth = row['depths']
    midtime = row['midtimes']
    period=  row['periods']
    
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = row['midtimes']                   #time of inferior conjunction
    params.per =  row['periods']                 #orbital period
    params.rp = row['depths']                     #planet radius (in units of stellar radii)
    stellar_radius = row['stellar_radius']
    semimaj = ((((7.496*(10**(-6)))*(period**2))**(1/3))*215.032)/stellar_radius #calc a based on period, and in terms of host star radius
    params.a = semimaj                    #semi-major axis (in units of stellar radii)
    params.inc = 89.                      #orbital inclination (in degrees)
    params.ecc = 0.                       #eccentricity
    params.w = 90.               
    params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    
    ###################################
    ##clean after injecting and proceed
    ###################################
    data = []
    for s in inject_lc:    
        datum =  eleanor.TargetData(s, height=15, width=15, bkg_size=31, aperture_mode='small', do_psf=False, do_pca=True, regressors='corner')
        q = datum.quality == 0

        lc_orig = datum.to_lightkurve()
        lc_orig = lc_orig.copy().normalize() ##MAKE SURE TO NORMALIZE BEFORE INJECTING else you add tiny number to raw flux val

        # Define the times at which to evaluate the fake transit
        t=lc_orig.time.value

        # Create the batman transit model
        m = batman.TransitModel(params, t)

        # Generate the fake light curve transit
        injected_model = m.light_curve(params)

        #plt.plot(t, injected_model)
        #plt.xlabel("Time from central transit")
        #plt.ylabel("Relative flux")
        #plt.show()


        # Inject the fake transit into the real data
        injected_flux = lc_orig.flux.value + injected_model - 1.0
        #injected_flux = lc_orig.flux.value * injected_model

        lc_injected=lc_orig.copy()
        lc_injected.flux = injected_flux
        lc_injected= lc_injected.copy() #we need this, it makes sure the flux u.quantity works with my math, otherwise if you do .remove_outliers in next step its fine



        lc_injected = lc_injected.remove_outliers(sigma=6).remove_nans()#.normalize().remove_nans()
        ##lc_orig = lc_orig.remove_outliers(sigma=6).remove_nans()#.normalize().remove_nans()


        endtime = max(lc_injected.time.value)-1
        starttime = min(lc_injected.time.value)+1
        lc_injected = lc_injected[(lc_injected['time'].value>starttime) & (lc_injected['time'].value<endtime)]
        lc_injected = lc_injected.remove_nans()

        ##endtime = max(lc_orig.time.value)-1
        ##starttime = min(lc_orig.time.value)+1
        ##lc_orig = lc_orig[(lc_orig['time'].value>starttime) & (lc_orig['time'].value<endtime)]
        ##lc_orig = lc_orig.remove_nans()



        ############################################################################
        gapsize = (lc_injected.time.value[1::] - lc_injected.time.value[:-1:])
        dt = np.median(gapsize)
        lc_injected['gapsize']=np.append(gapsize,0)

        lc_injected['index'] = np.arange(len(lc_injected))
        #cleanlc.add_index('index')

        for row in lc_injected:
            if row['gapsize']>0.5:
                idmax = row['index']+50 #removes 3.3 hrs of 200sec cadence
                idmin = row['index']-49
                lc_injected['flux'][idmin:idmax]=np.nan
        lc_injected = lc_injected.remove_nans()
        lc_injected['index'] = np.arange(len(lc_injected)) #reindex to avoid issues with removing points


        ############################################################################
        '''
        #only for diagnostic prints
        gapsize = (lc_orig.time.value[1::] - lc_orig.time.value[:-1:])
        dt = np.median(gapsize)
        lc_orig['gapsize']=np.append(gapsize,0)

        lc_orig['index'] = np.arange(len(lc_orig))
        #cleanlc.add_index('index')

        for row in lc_orig:
            if row['gapsize']>0.5:
                idmax = row['index']+50 #removes 3.3 hrs of 200sec cadence
                idmin = row['index']-49
                lc_orig['flux'][idmin:idmax]=np.nan
        lc_orig = lc_orig.remove_nans()
        lc_orig['index'] = np.arange(len(lc_orig)) #reindex to avoid issues with removing points
        '''


        ###############################
        # cut things that spike upwards
        ###############################
        sigma_upper = 4
        std = np.std(lc_injected['flux'])
        #print(std)
        for row in lc_injected:
            if row['flux']>(1 + (sigma_upper * std)):
                #print(row['flux'])
                idmax = row['index']+20
                idmin = row['index']-19
                lc_injected['flux'][idmin:idmax]=np.nan    
        lc_injected = lc_injected.remove_nans()
        lc_injected['index'] = np.arange(len(lc_injected)) #reindex to avoid issues with removing points


        '''
        ###############################
        # cut things that spike upwards
        ###############################
        sigma_upper = 4
        std = np.std(lc_orig['flux'])
        #print(std)
        for row in lc_orig:
            if row['flux']>(1 + (sigma_upper * std)):
                #print(row['flux'])
                idmax = row['index']+20
                idmin = row['index']-19
                lc_orig['flux'][idmin:idmax]=np.nan    
        lc_orig = lc_orig.remove_nans()
        lc_orig['index'] = np.arange(len(lc_orig)) #reindex to avoid issues with removing points
        '''



        lc_injected = lc_injected[(lc_injected['flux']<1.20)&(lc_injected['flux']>0.70)].remove_nans()#.bin(time_bin_size=binsizedays)
        #noinject = noinject.remove_nans().bin(time_bin_size=binsizedays)
        lc_injected['index'] = np.arange(len(lc_injected))#reindex to avoid issues with removing points

        '''
        #################################################
        # cut below expected signal of a 3.5 R_jup planet
        #################################################
        #lower_lim = 0.5

        #lower_lim = (1-(((3.5/9.73116)/stellar_radius)**2))
        lower_lim = (1-(((3/9.73116)/stellar_radius)**2))
        for row in lc_injected:
            if row['flux']<lower_lim:
                idmax = row['index']+2
                idmin = row['index']-1
                lc_injected['flux'][idmin:idmax]=np.nan

        '''
        lc_injected = lc_injected.bin(time_bin_size=binsizedays).remove_nans()
        ##lc_orig = lc_orig.bin(time_bin_size=binsizedays).remove_nans()

        data.append(lc_injected)
        ##odata.append(lc_orig)


    collect = lk.LightCurveCollection(data)
    ##ocollect = lk.LightCurveCollection(odata)

    lc_injected = collect.stitch(lambda x: x.remove_nans())

    
    #fig,axs=plt.subplots(3,1,figsize=(10,10))
    planetrad = depth*stellar_radius * 9.73116 #convert the solar radii to jupiter radii
    #lc_injected.scatter(ax=axs[0],s=25,color='r',label='injected transit signal')
    #normalized_corrected.scatter(ax=axs[0],s=25)
    
    period_grid = np.linspace(0.6, 10, 25000)
    bls = lc_injected.to_periodogram(method='bls', period=period_grid, frequency_factor=700);
    #bls.plot(ax=axs[1],label=f'best p = {bls.period_at_max_power:.2f}');
    planet_b_period = bls.period_at_max_power
    planet_b_t0 = bls.transit_time_at_max_power
    planet_b_dur = bls.duration_at_max_power
    #lc_injected.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axs[2],label='')

    blsorig = noinject_lc.to_periodogram(method='bls', period=period_grid, frequency_factor=700);
    origplanet_b_period = blsorig.period_at_max_power
    detectorig = abs((origplanet_b_period.value/period)-round(origplanet_b_period.value/period))

    detect = abs((planet_b_period.value/period)-round(planet_b_period.value/period))

    #axs[0].set_title(f'Rplanet(Rjup Radii) = {planetrad:.4f}, Midtime = {midtime:.2f}, Period = {period:.2f},operiod = {origplanet_b_period:.2f}')


    #print(f'detectval={detect}')
    accept_thresh=0.05
    if detectorig < accept_thresh:
        found = 0
    elif detect < accept_thresh:
        found = 1 #"True"
    else:
        found = 0 #"False"

    #print(f'depth: {planetrad:.4f}              actual:{period:.4f}             calc:{planet_b_period:.4f}    Recover?:{found}')
    #Rplanet.append(planetrad)
    #Pinject.append(period)
    #Pdetermine.append(planet_b_period.value)
    #recover.append(found)
    Rplanet=planetrad
    Pinject=period
    Pdetermine=planet_b_period.value
    recover=found


    #output_table['Rplanet'] = Rplanet
    #output_table['Pinject'] = Pinject
    #output_table['Pdetermine'] = Pdetermine
    #output_table['recover?'] = recover

    #return output_table
    return Rplanet, Pinject, Pdetermine, recover






def vary_paramsmult(func, star, df,stellar_radius=1.325, trials=10000, num_processes=1):
    t0 = time.time()
    
    noinjectstar = star.copy()
    #injectstar = star.copy()
    
    data = []
    for s in noinjectstar:  
        datum =  eleanor.TargetData(s, height=15, width=15, bkg_size=31, aperture_mode='small', do_psf=False, do_pca=True, regressors='corner')
        q = datum.quality == 0

        lc_orig = datum.to_lightkurve()
        lc_orig = lc_orig.normalize() ##MAKE SURE TO NORMALIZE BEFORE INJECTING else you add tiny number to raw flux val
        #print(lc_orig)
        lc_orig= lc_orig.copy() #we need this, it makes sure the flux u.quantity works with my math, otherwise if you do .remove_outliers in next step its fine
        #print(lc_orig)
        #lc_injected.scatter(c='red')
        #lc_orig.scatter()
        ###xtime = lc_injected['time'].copy()
        ###yflux = lc_injected['flux'].copy()
        ###plt.scatter(xtime,yflux,c='r')
        ###plt.scatter(lc_orig['time'],lc_orig['flux'])
        ###plt.show()

        lc_orig = lc_orig.remove_outliers(sigma=6).remove_nans()#.normalize().remove_nans()

        endtime = max(lc_orig.time.value)-1
        starttime = min(lc_orig.time.value)+1
        lc_orig = lc_orig[(lc_orig['time'].value>starttime) & (lc_orig['time'].value<endtime)]
        lc_orig = lc_orig.remove_nans()

        gapsize = (lc_orig.time.value[1::] - lc_orig.time.value[:-1:])
        dt = np.median(gapsize)
        lc_orig['gapsize']=np.append(gapsize,0)

        lc_orig['index'] = np.arange(len(lc_orig))
        #cleanlc.add_index('index')

        for row in lc_orig:
            if row['gapsize']>0.5:
                idmax = row['index']+50 #removes 3.3 hrs of 200sec cadence
                idmin = row['index']-49
                lc_orig['flux'][idmin:idmax]=np.nan
        lc_orig = lc_orig.remove_nans()
        lc_orig['index'] = np.arange(len(lc_orig)) #reindex to avoid issues with removing points

        ###############################
        # cut things that spike upwards
        ###############################
        sigma_upper = 4
        std = np.std(lc_orig['flux'])
        #print(std)
        for row in lc_orig:
            if row['flux']>(1 + (sigma_upper * std)):
                #print(row['flux'])
                idmax = row['index']+20
                idmin = row['index']-19
                lc_orig['flux'][idmin:idmax]=np.nan    
        lc_orig = lc_orig.remove_nans()
        lc_orig['index'] = np.arange(len(lc_orig)) #reindex to avoid issues with removing points



        lc_orig = lc_orig[(lc_orig['flux']<1.20)&(lc_orig['flux']>0.70)].remove_nans()#.bin(time_bin_size=binsizedays)
        lc_orig['index'] = np.arange(len(lc_orig))#reindex to avoid issues with removing points
        '''
        #################################################
        # cut below expected signal of a 3.5 R_jup planet
        #################################################
        #lower_lim = 0.5

        #lower_lim = (1-(((3.5/9.73116)/stellar_radius)**2))
        lower_lim = (1-(((3/9.73116)/stellar_radius)**2))
        for row in lc_orig:
            if row['flux']<lower_lim:
                idmax = row['index']+2
                idmin = row['index']-1
                lc_orig['flux'][idmin:idmax]=np.nan

        '''
        #unbindata.append(lc_orig.remove_nans())
        lc_orig = lc_orig.bin(time_bin_size=binsizedays).remove_nans()
        lc_orig = lc_orig.remove_nans()

        data.append(lc_orig)



    collect = lk.LightCurveCollection(data)
    noinject = collect.stitch(lambda x: x.remove_nans()) #no need to normalize a 2nd time

    noinject.scatter()
    
    def init():
        global inject_lc #lc to have planet injected
        inject_lc = noinjectstar
        global noinject_lc #reduced lightcurve of star without injection
        noinject_lc = noinject.copy()
    

    with mp.Pool(num_processes,initializer=init,maxtasksperchild=10) as pool:
        ##binned_lc.scatter()
        ##noinject.scatter()
        #print(output_table['depth'])
        #rad_min = 0.02115/stellar_radius #now in R_hoststar
        ####rad_min = 0.0616576/stellar_radius #now in R_hoststar for a 0.6 Jupiter radii planet
        rad_min = 0.14/stellar_radius #now in R_hoststar for a 0.6 Jupiter radii planet
        rad_max = 0.2/stellar_radius #now in R_hoststar for a 2 Jupiter radii planet

        depths = np.random.uniform(rad_min, rad_max, trials)  # random transit depths to inject
        midtimes = np.random.uniform(min(noinject.time.value), max(noinject.time.value), trials)  # mid-transit times to inject if you want
        ####periods = np.random.uniform(0.6,10,trials) # periods to inject
        periods = np.random.uniform(0.6,5,trials) # periods to inject
        
        df['depths'] = depths
        df['midtimes'] = midtimes
        df['periods'] = periods
        df['stellar_radius'] = stellar_radius
        #print(df)
    
    # we need a sequence to pass pool.map; this line creates a generator (lazy iterator) of columns

        seq = [(idx,row) for idx,row in df.iterrows()]

        # pool.map returns results as a list
        print('start the pool')
        results_list = pool.map(func, seq)

        # return list of processed columns, concatenated together as a new dataframe
        #return pd.concat(results_list, axis=0)
        #print(results_list)
        df[['Rplanet','Pinject','Pdetermine','recover?']] = results_list
    t1 = time.time()
    total = t1-t0
    print(f'Finished in {total}s')
    return df



#stellar_radius = 0.943
#tic = 77463924
#Tmag = 16.931
#star = eleanor.multi_sectors(sectors='all',tic=tic)
#input_lc_file = star
#n_trials = 1000
#n_processes = 96
#output_table_lcbin = pd.DataFrame()
#output_table_lcbin77463924 = vary_paramsmult(injectplanetbeforestitch,input_lc_file, output_table_lcbin, stellar_radius=stellar_radius, trials=n_trials, num_processes=n_processes)

