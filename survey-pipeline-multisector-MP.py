from tglc.quick_lc import tglc_lc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from astropy.modeling.models import BlackBody
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
from astropy.modeling import models
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import ascii
from astropy import table
from astropy.io.votable import parse
from astropy.convolution import convolve, Box1DKernel
from lightkurve.correctors import PLDCorrector
from astropy.io import fits
import lightkurve as lk
from astroquery.mast import Observations
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # Reselect Data Release 3, default
import multiprocessing as mp
import time
from pylab import *
from IPython.display import Image,display
import math
import batman
import os
import glob
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120

binsizedays = 30/60/24



##################################################################################################
## Pipeline designed to take a catalog of stars, pull all TESS Sectors, reduce the light curves, # 
## and search for periodic signals in the reduced light curves.                                  #
#                                                                                                #
## Outputs:                                                                                      #
#    #reduced stitched light curves                                                              #
#    #plot showing light curve, periodogram, folded lightcurve, and bls model                    #
#    #median scatter for lightcurves                                                             #
#    #period and transit duration for peak signal in periodogram                                 #
##################################################################################################

######################################################################################################
# Required Exoplanet Packages:                                                                       #
    #lightkurve                                                                                      #
    #TGLC                                                                                         #
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


savepath = '/orange/jasondittmann/wschap/streams/sagittarius/tglcearly/'


def periodsearch(arg):
    idx,row = arg
    #normalized_corrected = lcbin

    stellar_radius = row['Rad']
    #print(stellar_radius)
    tic = row['TIC']
    print(tic)
    Tmag = row['Tmag']

    if os.path.exists(f'{savepath}{tic}-reduced_lc.fits'):
        print(f'{tic} already fit, moving on to next star')
        starvalues = pd.read_csv(f'{savepath}{tic}-resulttable.csv', sep=',', header = 0)
        #print(tic)
        fluxval=starvalues['rawflux'][0]
        medstdbin=starvalues['medstdbin'][0]
        found_period=starvalues['found_period'][0]
        expected_dur_hr=starvalues['expected_dur_hr'][0]
        calc_dur_hr=starvalues['calc_dur_hr'][0]
        dur_vet=starvalues['dur_vet'][0]
        Depth=starvalues['Depth'][0]
        Edepth=starvalues['Edepth'][0]
        Odepth=starvalues['Odepth'][0]
        Rplanet=starvalues['Rplanet'][0]
        even_Rplanet=starvalues['even_Rplanet'][0]
        odd_Rplanet=starvalues['odd_Rplanet'][0]
        peakpower=starvalues['peakpower'][0]
        n_harmonics=starvalues['n_harmonics'][0]
    else:
        #sector = row['init_sector']
        #binsizedays = 30/60/24
        print(f'beginning star {tic}')
        try:
            target = f'TIC {tic}'     # Target ID ('TOI 519') or coordinates ('ra dec')
            targetname = f'TIC-{tic}'
            local_directory = f'/orange/jasondittmann/wschap/streams/sagittarius/tglcdata/{targetname}/'    # directory to save all files
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
            lcmast = lk.io.tglc.read_tglc_lightcurve(filelocation[0],flux_column='cal_aper_flux',quality_bitmask='none') #convert TGLC file to lightkurve for analysis

            mastfile = fits.open(filelocation[0])
            lcmast['flux_err'] = mastfile[1].header['CAPE_ERR'] #lightkurve currently cant grab the errors, so I add them to the file manually here
            mastfile[1].header['CAPE_ERR']
            lcmast['centroid_col'] = 0.0 #tglc doesnt output this for some reason
            lcmast['centroid_row'] = 0.0 #tglc doesnt output this for some reason

            lcmast= lcmast.copy()
            #lcbin = lcmast.remove_outliers(sigma=6).bin(time_bin_size=binsizedays).remove_nans() #bin the data so all sectors agree

            lcbin = lcmast.remove_outliers(sigma=6).remove_nans() #bin the data so all sectors agree
            fluxval = np.average(lcbin['aperture_flux']).value #flux from aperature measure
            fluxval = float(fluxval)


            endtime = max(lcbin.time.value)-1
            starttime = min(lcbin.time.value)+1
            lcbin = lcbin[(lcbin['time'].value>starttime) & (lcbin['time'].value<endtime)]
            lcbin = lcbin.remove_nans()

            gapsize = (lcbin.time.value[1::] - lcbin.time.value[:-1:])
            dt = np.median(gapsize)
            lcbin['gapsize']=np.append(gapsize,0)

            lcbin['index'] = np.arange(len(lcbin))
            #cleanlc.add_index('index')

            for row in lcbin:
                if row['gapsize']>0.5:
                    idmax = row['index']+50 #removes 3.3 hrs of 200sec cadence
                    idmin = row['index']-49
                    lcbin['flux'][idmin:idmax]=np.nan
            lcbin = lcbin.remove_nans()
            lcbin['index'] = np.arange(len(lcbin)) #reindex to avoid issues with removing points

            ###############################
            # cut things that spike upwards
            ###############################
            sigma_upper = 4
            std = np.std(lcbin['flux'])
            #print(std)
            for row in lcbin:
                if row['flux']>(1 + (sigma_upper * std)):
                    #print(row['flux'])
                    idmax = row['index']+20
                    idmin = row['index']-19
                    lcbin['flux'][idmin:idmax]=np.nan    
            lcbin = lcbin.remove_nans()
            lcbin['index'] = np.arange(len(lcbin)) #reindex to avoid issues with removing points



            lcbin = lcbin[(lcbin['flux']<1.20)&(lcbin['flux']>0.70)].remove_nans()#.bin(time_bin_size=binsizedays)
            lcbin['index'] = np.arange(len(lcbin))#reindex to avoid issues with removing points

            lcbin = lcbin.bin(time_bin_size=binsizedays).remove_nans()

            #lcbin = lcbin.remove_nans()

            #############################################################
            #cant save the cleaned output as of now, not at all sure why
            #############################################################
            lcbin.to_fits(path=f'{savepath}{tic}-reduced_lc.fits', overwrite=True)


            ###sagstreammore.loc[index,'rawflux'] = fluxval

            medstdbin = np.median(np.abs(lcbin['flux']-np.median(lcbin['flux'])))
            #print(f'scatter = {medstdbin}')
            #####sagstreammore.loc[index,'medstdbin'] = medstdbin


            #lets search for periodic signals

            period = np.linspace(0.6, 10, 25000)
            periodogram = lcbin.to_periodogram(method='bls', period=period, frequency_factor=700);
            #ppmval = lcbin.estimate_cdpp()

            planet_b_period = periodogram.period_at_max_power
            planet_b_t0 = periodogram.transit_time_at_max_power
            planet_b_dur = periodogram.duration_at_max_power



            fig,axes = plt.subplots(3,2, figsize=(15,15))

            ax1 = lcbin.scatter(ax=axes[0,0],color='black')
            ax1.set_title(f'Light Curve of TIC {tic}')
            ax2 = periodogram.plot(ax=axes[0,1],color='black')
            ax2.set_title(f'BLS Periodigram')

            ax3 = lcbin.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axes[1,0],s=12)

            ax3.set_title(f'TIC {tic} folded at Period={planet_b_period:.2f}')
            ##ax3.set_xlim(-0.75, 0.75);


            planet_b_model = periodogram.get_transit_model(period=planet_b_period,
                                                   transit_time=planet_b_t0,
                                                   duration=planet_b_dur)

            #Now we should get an idea of planet radius while we have a depth
            trandepth = planet_b_model['flux'].min()
            #print(f'Recovered Transit Depth = {trandepth}')
            #####sagstreammore.loc[index,'Depth'] = trandepth
            Depth = trandepth
            Rplanet = (np.sqrt(1-trandepth)*stellar_radius)* 9.73116 #in Rjupiter
            #####sagstreammore.loc[index,'Rplanet'] = Rplanet
            Rplanet = Rplanet
            #print(f'planet radius = {Rplanet}')

            ax4 = lcbin.fold(planet_b_period, planet_b_t0).scatter(ax=axes[1,1],s=12)
            planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=axes[1,1], c='r', lw=2)
            ax4.set_title(f'Full Transit Model: Depth={trandepth:.2f}, Rplanet={Rplanet:.2f}')
            ##ax4.set_xlim(-0.2, 0.2);


            ########################
            #even odd statistics
            ########################
            f = lcbin.fold(planet_b_period, planet_b_t0)

            oddtransit = f[f.odd_mask]
            eventransit = f[f.even_mask]

            odd_periodogram = lcbin[f.odd_mask].to_periodogram(method='bls', period=period, frequency_factor=700);
            even_periodogram = lcbin[f.even_mask].to_periodogram(method='bls', period=period, frequency_factor=700);


            odd_planet_b_period = odd_periodogram.period_at_max_power
            odd_planet_b_t0 = odd_periodogram.transit_time_at_max_power
            odd_planet_b_dur = odd_periodogram.duration_at_max_power

            even_planet_b_period = even_periodogram.period_at_max_power
            even_planet_b_t0 = even_periodogram.transit_time_at_max_power
            even_planet_b_dur = even_periodogram.duration_at_max_power


            odd_model = odd_periodogram.get_transit_model(period=odd_planet_b_period,
                                                   transit_time=odd_planet_b_t0,
                                                   duration=odd_planet_b_dur)

            even_model = even_periodogram.get_transit_model(period=even_planet_b_period,
                                                   transit_time=even_planet_b_t0,
                                                   duration=even_planet_b_dur)


            odd_trandepth = odd_model['flux'].min()
            Odepth=odd_trandepth
            even_trandepth = even_model['flux'].min()
            Edepth=even_trandepth

            odd_Rplanet = (np.sqrt(1-odd_trandepth)*stellar_radius)* 9.73116 #in Rjupiter

            even_Rplanet = (np.sqrt(1-even_trandepth)*stellar_radius)* 9.73116 #in Rjupiter



            ax5 = oddtransit.scatter(ax=axes[2,0], lw=2)
            ax5.set_ylim(ax4.get_ylim())
            odd_model.fold(odd_planet_b_period, odd_planet_b_t0).plot(ax=axes[2,0], c='r', lw=2)
            ax5.set_title(f'Odd Transits: Depth={odd_trandepth:.3}, Rplanet={odd_Rplanet:.3}')

            ax6 = eventransit.scatter(ax=axes[2,1],s=12)
            ax6.set_ylim(ax4.get_ylim())
            even_model.fold(even_planet_b_period, even_planet_b_t0).plot(ax=axes[2,1], c='r', lw=2)
            ax6.set_title(f'Even Transits: Depth={even_trandepth:.3}, Rplanet={even_Rplanet:.3}')

            fig.tight_layout()
            plt.savefig(f'{savepath}{tic}-reduced_lc_plot.jpg',bbox_inches='tight')
            ##progress=[(index+1)/surveysize]
            ##outtable['progress']=progress
            ##outtable.to_csv('/orange/jasondittmann/wschap/streams/sagittarius/cleanedlk/progress.csv')



            #find the calculated transit duration
            calcduration = planet_b_dur
            calhrduration = planet_b_dur.value*24*u.hr
            calc_dur_hr = calhrduration.value
            #print(f'calculated duration is {calhrduration}, for TIC {tic}')


            #now lets find the expected duration for a circular, edge on transit. If its longer than we calculate were good
            #if its shorter, periodogram is likely finding a non planet signal 

            periodcalc = planet_b_period #days (we use the periodogram period)
            found_period = periodcalc.value
            impact_param = 0.00 #equator cross (longest transit length)
            #stellar_radius = Rstar #taken from catalog, in Rsol
            semimaj = ((((7.496*(10**(-6)))*(periodcalc.value**2))**(1/3))*215.032)/stellar_radius #calc a based on period, and in terms of host star radius (a/R* au)
            scaled_planet_radius = np.sqrt(planet_b_model['flux'].min()) #Rp/R*

            expectduration = Tt(periodcalc, impact_param, semimaj, scaled_planet_radius) #find expected duration
            expectdurhour = expectduration.value*24*u.hr
            expected_dur_hr = expectdurhour.value



            if expectduration.value > calcduration.value:
                #print(f'this could work')
                #####sagstreammore.loc[index,'dur_vet'] = 1
                dur_vet = 1
            else:
                #print(f'this is a bit large')
                #####sagstreammore.loc[index,'dur_vet'] = 0
                dur_vet = 0

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
            #print(f'peak power = {peakpower}')
            #####sagstreammore.loc[index,'peakpower'] = peakpower
            #print(peakpower)

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
            #####sagstreammore.loc[index,'n_harmonics'] = counter
            #print(f'number of possible harmonics = {counter}')
            n_harmonics = counter
            #print(blsdfshort)




        #if all else fails, just move on to the next star for now, troubleshoot this at a later time 
        except:
            #####sagstreammore.loc[index,'medstdbin'] = 999
            print(f'Error occurred with TIC {tic}')
            fluxval=999
            medstdbin=999
            found_period=999 
            expected_dur_hr=999
            calc_dur_hr=999
            dur_vet=999
            Depth=999
            Edepth=999 
            Odepth=999 
            Rplanet=999
            even_Rplanet=999 
            odd_Rplanet=999
            peakpower=999
            n_harmonics=999


        #output_table['Rplanet'] = Rplanet
        #output_table['Pinject'] = Pinject
        #output_table['Pdetermine'] = Pdetermine
        #output_table['recover?'] = recover

        ##########################################################################
        # the MP sometimes freezes, maybe a memory leak? Either way
        # Im addint a save of the important outputs to each star that
        # can later be combined into a larger results array since the current
        # formatting only outputs that final results df upon successful completion
        # of the MP run
        ##########################################################################
        '''
        ['TIC':tic, 
         'stellar_radius':stellar_radius, 
         'Tmag':Tmag,
         'rawflux':fluxval, 
         'medstdbin':medstdbin, 
         'found_period':found_period, 
         'expected_dur_hr':expected_dur_hr, 
         'calc_dur_hr':calc_dur_hr, 
         'dur_vet':dur_vet, 
         'Depth':Depth, 
         'Rplanet':Rplanet, 
         'peakpower':peakpower, 
         'n_harmonics':n_harmonics]
         '''

        starcols = {'TIC':tic, 
         'stellar_radius':stellar_radius, 
         'Tmag':Tmag,
         'rawflux':fluxval, 
         'medstdbin':medstdbin, 
         'found_period':found_period, 
         'expected_dur_hr':expected_dur_hr, 
         'calc_dur_hr':calc_dur_hr, 
         'dur_vet':dur_vet, 
         'Depth':Depth,
         'Edepth':Edepth,
         'Odepth':Odepth,
         'Rplanet':Rplanet, 
         'even_Rplanet':even_Rplanet,
         'odd_Rplanet': odd_Rplanet,
         'peakpower':peakpower, 
         'n_harmonics':n_harmonics}
        staroutput = pd.DataFrame(data=starcols, index=[0])
        #print(staroutput)
        staroutput.to_csv(f'{savepath}{tic}-resulttable.csv',index=False)

    #return output_table
    return  tic, stellar_radius, Tmag, fluxval, medstdbin, found_period, expected_dur_hr, calc_dur_hr, dur_vet, Depth, Edepth, Odepth, Rplanet, even_Rplanet, odd_Rplanet, peakpower, n_harmonics
    ###
    #sometimes the flux output is broken, temp fix was to force it float, if that fails use this return instead
    ###
    #return  tic, stellar_radius, Tmag, medstdbin, found_period, expected_dur_hr, calc_dur_hr, dur_vet, Depth, Rplanet, peakpower, n_harmonics






def vary_paramsmult(func, df, num_processes=1):
    t0 = time.time()

    #####with mp.Pool(num_processes,initializer=init,maxtasksperchild=10) as pool:
    with mp.Pool(num_processes,maxtasksperchild=1) as pool:    

        #sagstreammore = pd.read_csv('/blue/jasondittmann/wschap/streams/sagittarius/earlytglcscatter-slurm-bgestimate.csv', sep=',', header = 0)
        #sagstreammore = sagstreammore.head(5)
        
        #################################################################################
        # Open df with star data and cut down to stars we can detect jupiteres around
        #################################################################################
        sagstreamearly = pd.read_csv('/blue/jasondittmann/wschap/streams/sagittarius/ramos22_radcut3_gmagcut18_probcut50_neighbors_initsector_meddev_mproc-tess-observed-earlysector.csv', sep=',', header = 0)
        sagstreamearlyrad = sagstreamearly[sagstreamearly['Rad']<2]
        sagstreammore = sagstreamearlyrad[sagstreamearlyrad['Tmag']<17]
        #sagstreammore = sagstreammore.head(4)
        #sagstreammore = sagstreamearly[sagstreamearly['TIC'].isin(priorityticsearly)]
        #sagstreammore = sagstreammid[sagstreammid['TIC'].isin(priorityticsmid)]
        #sagstreammore = sagstreamlate[sagstreamlate['TIC'].isin(priorityticslate)]
        #sagstreammore = sagstreammore.head(10)
        
        surveysize = sagstreammore.shape[0]
        outtable = pd.DataFrame()
        t0 = time.time()
    
    # we need a sequence to pass pool.map; this line creates a generator (lazy iterator) of columns

        seq = [(idx,row) for idx,row in sagstreammore.iterrows()]

        # pool.map returns results as a list
        print('start the pool')
        results_list = pool.map(func, seq)

        # return list of processed columns, concatenated together as a new dataframe
        #return pd.concat(results_list, axis=0)
        #print(results_list)
        df[['TIC', 'stellar_radius', 'Tmag','rawflux', 'medstdbin', 'found_period', 
            'expected_dur_hr', 'calc_dur_hr', 'dur_vet', 'Depth', 'Edepth','Odepth','Rplanet','even_Rplanet', 'odd_Rplanet', 'peakpower', 'n_harmonics']] = results_list

    t1 = time.time()
    total = t1-t0
    print(f'Finished in {total}s')
    return df



n_processes = 32
output_table_lcbin = pd.DataFrame()
outtable = vary_paramsmult(periodsearch, output_table_lcbin, num_processes=n_processes)
outtable.TIC = outtable.TIC.astype(int)
outtable.to_csv(f'{savepath}resulttable.csv',index=False)
