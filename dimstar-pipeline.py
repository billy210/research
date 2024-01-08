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


plt.rcParams['figure.dpi'] = 120





##################################################################################################
## Pipeline designed to take a catalog of stars, pull all TESS Sectors, reduce the light curves, # 
## and search for periodic signals in the reduced light curves.                                  #
## Forks include:                                                                                #
#    #TIC flux used to infer crowding/contamination                                              #
#        # -if dim as expected, use eleanor                                                      #
#        # -if abnormally bright, use TGLC to deal with crowding better                          #
#    #Rejecting having found the TESS period                                                     #
#        # -if period is found to be close to 13.7 (TESS orbital period)                         #
#        #  mask out the "period" and try again                                                  #
## Outputs:                                                                                      #
#    #reduced stitched light curves                                                              #
#    #plot showing light curve, periodogram, folded lightcurve, and bls model                    #
#    #median scatter for lightcurves                                                             #
#    #period and transit duration for peak signal in periodogram                                 #
##################################################################################################

######################################################################################################
# Required Exoplanet Packages:                                                                       #
    #lightkurve                                                                                      #
    #eleanor                                                                                         #
    #TGLC (TESS Gaia Light Curve)                                                                    #
######################################################################################################

#########################################################################################
# Ben Capistrant wrote Tt() function below, any questions should go to him              #
# function finds transit time given:                                                    #
#    #period [days] (we use the periodogram period)                                     #
#    #impact_param = 0.00 (equator cross for longest transit length)                    #
#    #stellar_radius [Rsol]                                                             #
#    #semimaj = #calc based on period, and in terms of host star radius [a/R*]          #
#    #scaled_planet_radius Rp/R*                                                        #
#########################################################################################
def Tt(period, impact_param, scaled_semimajor_axis, scaled_planet_radius):
    return ((period / np.pi)
            * np.arcsin(np.sqrt(((1 + scaled_planet_radius) ** 2 - impact_param ** 2)
                                / (scaled_semimajor_axis ** 2 - impact_param ** 2))))
#########################################################################################
#########################################################################################


sagstreammore = pd.read_csv('/blue/jasondittmann/wschap/streams/sagittarius/earlytglcscatter-slurm-bgestimate.csv', sep=',', header = 0)
sagstreammore = sagstreammore.head(2)
surveysize = sagstreammore.shape[0]
outtable = pd.DataFrame()
t0 = time.time()

#check the star in eleanor, and get an idea of the flux. Too high will imply contamination from either
#brights stars or a crowded field. This will allow us to decide between using eleanor or TGLC for analysis.


for index, row in sagstreammore.iterrows():
    stellar_radius = row['Rad']
    tic = row['TIC']
    Tmag = row['Tmag']
    #sector = row['init_sector']
    #binsizedays = 30/60/24
    print(f'beginning star {tic}')
    try:
        star = eleanor.multi_sectors(sectors='all',tic=tic)
        data = []

        for s in star:   
            datum =  eleanor.TargetData(s, height=15, width=15, bkg_size=31, aperture_mode='small', do_psf=False, do_pca=True, regressors='corner')
            q = datum.quality == 0

            lc = datum.to_lightkurve()
            lc = lc.normalize().remove_nans()
            data.append(lc)
        fluxval = np.median(datum.raw_flux)

        print(f'flux founds to be {fluxval}')

        #see if the flux is <95, which is close to the estimated flux of a 15.5mag star, if true, continue analysis
        if fluxval < 95:
            print('continuing with eleanor')

            data = []
            for s in star:  
                datum =  eleanor.TargetData(s, height=15, width=15, bkg_size=31, aperture_mode='small', do_psf=False, do_pca=True, regressors='corner')
                q = datum.quality == 0

                lc_orig = datum.to_lightkurve()
                lc_orig = lc_orig.normalize() ##MAKE SURE TO NORMALIZE BEFORE INJECTING else you add tiny number to raw flux val
                #print(lc_orig)
                lc_orig= lc_orig.copy() #we need this, it makes sure the flux u.quantity works with my math, otherwise if you do .remove_outliers in next step its fine


                lc_orig = lc_orig.remove_outliers(sigma=6).remove_nans()

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

                lc_orig = lc_orig.bin(time_bin_size=binsizedays).remove_nans()
                lc_orig = lc_orig.remove_nans()

                data.append(lc_orig)

            collect = lk.LightCurveCollection(data)
            lcbin = collect.stitch(lambda x: x.remove_nans()) #no need to normalize a 2nd time

            lcbin.scatter()



            lcbin.to_fits(path=f'/orange/jasondittmann/wschap/streams/sagittarius/cleanedlk/{tic}-reduced_lc.fits', overwrite=True)
            sagstreammore.loc[index,'rawflux'] = fluxval

            medstdbin = np.median(np.abs(lcbin['flux']-np.median(lcbin['flux'])))
            #print(f'scatter = {medstdbin}')
            sagstreammore.loc[index,'medstdbin'] = medstdbin


            #lets search for periodic signals

            period = np.linspace(0.6, 10, 25000)
            periodogram = lcbin.to_periodogram(method='bls', period=period, frequency_factor=700);
            #ppmval = lcbin.estimate_cdpp()

            planet_b_period = periodogram.period_at_max_power
            planet_b_t0 = periodogram.transit_time_at_max_power
            planet_b_dur = periodogram.duration_at_max_power



            fig,axes = plt.subplots(2,2, figsize=(15,10))

            ax1 = lcbin.scatter(ax=axes[0,0],color='black')
            ax1.set_title(f'Light Curve of TIC {tic}')
            ax2 = periodogram.plot(ax=axes[0,1],color='black')
            ax2.set_title(f'Periodigram of TIC {tic}')

            ax3 = lcbin.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axes[1,0],s=12)

            ax3.set_title(f'TIC {tic} folded at {planet_b_period:.2f}')
            ax3.set_xlim(-0.75, 0.75);


            planet_b_model = periodogram.get_transit_model(period=planet_b_period,
                                                   transit_time=planet_b_t0,
                                                   duration=planet_b_dur)

            ax4 = lcbin.fold(planet_b_period, planet_b_t0).scatter(ax=axes[1,1],s=12)
            planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=axes[1,1], c='r', lw=2)
            ax4.set_xlim(-0.2, 0.2);
            fig.tight_layout()
            plt.savefig(f'/orange/jasondittmann/wschap/streams/sagittarius/cleanedlk/{tic}-reduced_lc_plot.jpg',bbox_inches='tight')
            progress=[(index+1)/surveysize]
            outtable['progress']=progress
            outtable.to_csv('/orange/jasondittmann/wschap/streams/sagittarius/cleanedlk/progress.csv')



            #find the calculated transit duration
            calcduration = planet_b_dur
            calhrduration = planet_b_dur.value*24*u.hr
            print(f'calculated duration is {calhrduration}, for TIC {tic}')


            #now lets find the expected duration for a circular, edge on transit. If its longer than we calculate were good
            #if its shorter, periodogram is likely finding a non planet signal 

            periodcalc = planet_b_period #days (we use the periodogram period)
            impact_param = 0.00 #equator cross (longest transit length)
            #stellar_radius = Rstar #taken from catalog, in Rsol
            semimaj = ((((7.496*(10**(-6)))*(periodcalc.value**2))**(1/3))*215.032)/stellar_radius #calc a based on period, and in terms of host star radius (a/R* au)
            scaled_planet_radius = np.sqrt(planet_b_model['flux'].min()) #Rp/R*

            expectduration = Tt(periodcalc, impact_param, semimaj, scaled_planet_radius) #find expected duration
            expectdurhour = expectduration.value*24*u.hr

            #print(f'Expected  duration is {expectduration}, or {expectdurhour}')
            #print(f'Expected duration in hrs {expectdurhour}')
            #print(f'Calculated duration in hrs {calhrduration}')
            sagstreammore.loc[index,'found_period'] = periodcalc.value
            sagstreammore.loc[index,'expected_dur_hr'] = expectdurhour.value
            sagstreammore.loc[index,'calc_dur_hr'] = calhrduration.value

            if expectduration.value > calcduration.value:
                print(f'this could work')
                sagstreammore.loc[index,'dur_vet'] = 1
            else:
                print(f'this is a bit large')
                sagstreammore.loc[index,'dur_vet'] = 0


            #Now we should get an idea of planet radius while we have a depth
            trandepth = planet_b_model['flux'].min()
            #print(f'Recovered Transit Depth = {trandepth}')
            sagstreammore.loc[index,'Depth'] = trandepth
            Rplanet = (np.sqrt(1-trandepth)*stellar_radius)* 9.73116 #in Rjupiter
            sagstreammore.loc[index,'Rplanet'] = Rplanet
            #print(f'planet radius = {Rplanet}')

            #####################
            # check for harmonics
            #####################
            blsdf = pd.DataFrame()
            blsdf['power'] = periodogram.power.value
            blsdf['period'] = periodogram.period.value
            blsdf['harmonic_math'] = (periodogram.period_at_max_power)/(blsdf['period'])
            blsdf = blsdf.sort_values(by='power', ascending=False).reset_index()
            blsdfshort = blsdf[0:10]
            peakpower = blsdf['power'][0]
            print(f'peak power = {peakpower}')
            sagstreammore.loc[index,'peakpower'] = peakpower

            counter=0
            for index2, row2 in blsdfshort.iterrows():
                if abs(((row2['harmonic_math']-2)/2)) <= 0.05:
                    #print('2x harmonic')
                    row2['harmonic_check'] = abs(((row2['harmonic_math']-2)/2))
                    counter = counter+1
                elif abs(((row2['harmonic_math']-0.5)/0.5)) <= 0.05:
                    #print('0.5 harmonic')
                    row2['harmonic_check'] = abs(((row2['harmonic_math']-0.5)/0.5))
                    counter = counter+1
                else:
                    #print('might be noise')
                    row2['harmonic_check'] = 999
                    pass
            sagstreammore.loc[index,'n_harmonics'] = counter
            print(f'number of possible harmonics = {counter}')
            #print(blsdfshort)
   
            
            
        #if the flux was too large, run it through TGLC, it takes longer, but does better in contaminated fields    
        else:
            print(f'est flux of {fluxval} too large and may contain significant contamination, using TGLC instead')
            target = f'TIC {tic}'     # Target ID ('TOI 519') or coordinates ('ra dec')
            targetname = f'TIC-{tic}'
            local_directory = f'/blue/jasondittmann/wschap/streams/sagittarius/tglcdata/{targetname}/'    # directory to save all files
            os.makedirs(local_directory, exist_ok=True)
            obs_table = tglc_lc(target=target, 
                    local_directory=local_directory, 
                    size=50, # FFI cutsize. Recommand at least 50 or larger for better performance. Cannot exceed 99. 
                             # Downloading FFI might take longer (or even cause timeouterror) for larger sizes. 
                    save_aper=False, # whether to save 5*5 pixels timeseries of the decontaminated images in fits file primary HDU
                    limit_mag=17, # the TESS magnitude lower limit of stars to output
                    get_all_lc=False, # whether to return all lcs in the region. If False, return the nearest star to the target coordinate
                    first_sector_only=True, # whether to return only lcs from the sector this target was first observed. 
                                            # If False, return all sectors of the target, but too many sectors could be slow to download.
                    sector=None, # If first_sector_only=True, sector will be ignored.
                                 # If first_sector_only=False and sector = None, return all observed sectors
                                 # If first_sector_only=False and sector = 1, return only sector 1. 
                                 # (Make sure only put observed sectors. All available sectors are printed in the sector table.)
                    prior=None)  # If None, does not allow all field stars to float. SUGGESTED for first use. 
                                 # If float (usually <1), allow field stars to float with a Gaussian prior with the mean 
                                 # at the Gaia predicted value the width of the prior value multiplied on the Gaia predicted value.) 


            filelocation = glob.glob(f'{local_directory}lc/*.fits')
            lcmast = lk.io.tglc.read_tglc_lightcurve(filelocation[0],quality_bitmask='none') #convert TGLC file to lightkurve for analysis

            mastfile = fits.open(filelocation[0])
            lcmast['flux_err'] = mastfile[1].header['CPSF_ERR'] #lightkurve currently cant grab the errors, so I add them to the file manually here

            lcbin = lcmast.remove_outliers(sigma=6).bin(time_bin_size=binsizedays).remove_nans() #bin the data so all sectors agree
            
            lcbin.to_fits(path=f'/blue/jasondittmann/wschap/streams/sagittarius/cleanedlk/{tic}-reduced_lc.fits', overwrite=True)
            
            #lcbin.scatter()

            medstdbin = np.median(np.abs(lcbin['flux']-np.median(lcbin['flux'])))
            #print(f'scatter = {medstdbin}')
            sagstreammore.loc[index,'medstdbin'] = medstdbin #save the result here
            
            
            
            #lets search for periodic signals
            
            period = np.linspace(1, 18, 50000)
            periodogram = lcbin.to_periodogram(method='bls', period=period, frequency_factor=500);
            
            planet_b_period = periodogram.period_at_max_power
            planet_b_t0 = periodogram.transit_time_at_max_power
            planet_b_dur = periodogram.duration_at_max_power
            #ppmval = lcbin.estimate_cdpp()


            if (planet_b_period.value > 13.4) & (planet_b_period.value <14.0):
                print('masking out the TESS period')

                # Create a cadence mask using the BLS parameters
                planet_b_mask = periodogram.get_transit_mask(period=planet_b_period,
                                                     transit_time=planet_b_t0,
                                                     duration=planet_b_dur)


                masked_lc = stitching[~planet_b_mask]


                period = np.linspace(1, 18, 50000)
                periodogram = masked_lc.to_periodogram(method='bls', period=period, frequency_factor=500);
                #ppmval = lcbin.estimate_cdpp()
                fig,axes = plt.subplots(2,2, figsize=(15,10))

                ax1 = masked_lc.scatter(ax=axes[0,0],color='black')
                ax1.set_title(f'Light Curve of TIC {tic}')
                ax2 = periodogram.plot(ax=axes[0,1],color='black')
                ax2.set_title(f'Periodigram of TIC {tic}')
                #plt.savefig(f'/blue/sarahballard/wschap/ariadne-results/mpfigures/{ticnum}.pdf')

                planet_b_period = periodogram.period_at_max_power
                planet_b_t0 = periodogram.transit_time_at_max_power
                planet_b_dur = periodogram.duration_at_max_power

                ax3 = masked_lc.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axes[1,0],s=12)

                ax3.set_title(f'TIC {tic} folded at {planet_b_period:.2f}')
                ax3.set_xlim(-0.75, 0.75);


                # Create a BLS model using the BLS parameters and plot resulting transit fit


                planet_b_model = periodogram.get_transit_model(period=planet_b_period,
                                                       transit_time=planet_b_t0,
                                                       duration=planet_b_dur)


                ax4 = masked_lc.fold(planet_b_period, planet_b_t0).scatter(ax=axes[1,1],s=12)
                planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=axes[1,1], c='r', lw=2)
                ax4.set_xlim(-0.2, 0.2);

                #print(f'period of {planet_b_period}')
                #print(f'transit duration of {planet_b_dur}')

                fig.tight_layout()
                plt.savefig(f'/blue/jasondittmann/wschap/streams/sagittarius/cleanedlk/{tic}-reduced_lc.jpg',bbox_inches='tight')
                progress=[(index+1)/surveysize]
                outtable['progress']=progress
                outtable.to_csv('/blue/jasondittmann/wschap/streams/sagittarius/cleanedlk/progress.csv')


            else:
                print(f'period of {planet_b_period} not TESS period')
                
                fig,axes = plt.subplots(2,2, figsize=(15,10))

                ax1 = lcbin.scatter(ax=axes[0,0],color='black')
                ax1.set_title(f'Light Curve of TIC {tic}')
                ax2 = periodogram.plot(ax=axes[0,1],color='black')
                ax2.set_title(f'Periodigram of TIC {tic}')



                ax3 = lcbin.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axes[1,0],s=12)

                ax3.set_title(f'TIC {tic} folded at {planet_b_period:.2f}')
                ax3.set_xlim(-0.75, 0.75);


                planet_b_model = periodogram.get_transit_model(period=planet_b_period,
                                                       transit_time=planet_b_t0,
                                                       duration=planet_b_dur)

                ax4 = lcbin.fold(planet_b_period, planet_b_t0).scatter(ax=axes[1,1],s=12)
                planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=axes[1,1], c='r', lw=2)
                ax4.set_xlim(-0.2, 0.2);
                
                fig.tight_layout()
                plt.savefig(f'/blue/jasondittmann/wschap/streams/sagittarius/cleanedlk/{tic}-reduced_lc.jpg',bbox_inches='tight')
                progress=[(index+1)/surveysize]
                outtable['progress']=progress
                outtable.to_csv('/blue/jasondittmann/wschap/streams/sagittarius/cleanedlk/progress.csv')



            #find the calculate transit duration
            calcduration = planet_b_dur
            calhrduration = planet_b_dur.value*24*u.hr
            #print(f'calculated duration is {calcduration}, or {calhrduration}')
            print(f'calculated duration is {calhrduration}, for TIC {tic}')


            #now lets find the expected duration for a circular, edge on transit. If its longer than we calculate were good
            #if its shorter, periodogram is likely finding a non planet signal 

            periodcalc = planet_b_period #days (we use the periodogram period)
            impact_param = 0.00 #equator cross (longest transit length)
            stellar_radius = Rstar #taken from catalog, in Rsol
            semimaj = ((((7.496*(10**(-6)))*(periodcalc.value**2))**(1/3))*215.032)/stellar_radius #calc a based on period, and in terms of host star radius (a/R* au)
            scaled_planet_radius = np.sqrt(planet_b_model['flux'].min()) #Rp/R*

            expectduration = Tt(periodcalc, impact_param, semimaj, scaled_planet_radius) #find expected duration
            expectdurhour = expectduration.value*24*u.hr

            #print(f'Expected  duration is {expectduration}, or {expectdurhour}')
            #print(f'Expected duration in hrs {expectdurhour}')
            sagstreammore.loc[index,'found_period'] = periodcalc.value
            sagstreammore.loc[index,'expected_dur_hr'] = expectdurhour.value
            sagstreammore.loc[index,'calc_dur_hr'] = calhrduration.value

            if expectduration.value > calcduration.value:
                print(f'this could work')
                sagstreammore.loc[index,'dur_vet'] = 1
            else:
                print(f'this is a bit large')
                sagstreammore.loc[index,'dur_vet'] = 0
    
    #if all else fails, just move on to the next star for now, troubleshoot this at a later time 
    except:
        sagstreammore.loc[index,'medstdbin'] = 999
        print(f'No lc was produced for TIC {tic}')
        continue





sagstreammore.to_csv('/blue/jasondittmann/wschap/streams/sagittarius/cleanedlk/resulttable.csv')
t1 = time.time()
total = t1-t0
totalmin = total/60
totalhour = totalmin/60
print(f'ran for {totalmin} minutes (that means {totalhour} hours)')
