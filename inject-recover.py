#Runs an injection recovery for a given light curve. Each iteration injects a different random period and planet radius, performs a periodogram analysis to try and find the period, 
#and outputs a 0 or 1 depending on whether the correct period was recovered or not. It is output into a pandas table column as ['recover?']. Plotting script is also included.
#####normalized_corrected: a processed light curve that will be used for the injection recovery
#####stellar_radius: the radius of the star in solar radii
#####trials: number of injected planets to try for the injected recovery. Recommend more than 1000 since the statistics are a bit iffy when a lower value is selected

import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
import batman
from astropy.timeseries import BoxLeastSquares


def vary_params(normalized_corrected, stellar_radius=1.325, trials=1000):
    output_table = pd.DataFrame()
    Rplanet = []
    Pinject = []
    Pdetermine = []
    recover = []
    
    #print(output_table['depth'])
    rad_min = 0.02115/stellar_radius #now in R_hoststar
    rad_max = 0.2/stellar_radius #now in R_hoststar
    
    depths = np.random.uniform(rad_min, rad_max, trials)  # random transit depths to inject
    midtimes = np.random.uniform(min(normalized_corrected.time.value), max(normalized_corrected.time.value), trials)  # mid-transit times to inject if you want
    periods = np.random.uniform(0.3,18,trials) # periods to inject

    for depth, midtime, period in zip(depths, midtimes, periods):
        params = batman.TransitParams()       #object to store transit parameters
        params.t0 = midtime                   #time of inferior conjunction
        params.per =  period                  #orbital period
        params.rp = depth                     #planet radius (in units of stellar radii)
        semimaj = ((((7.496*(10**(-6)))*(period**2))**(1/3))*215.032)/stellar_radius #calc a based on period, and in terms of host star radius
        params.a = semimaj                    #semi-major axis (in units of stellar radii)
        params.inc = 89.                      #orbital inclination (in degrees)
        params.ecc = 0.                       #eccentricity
        params.w = 90.               
        params.u = [0.1, 0.3]                #limb darkening coefficients [u1, u2]
        params.limb_dark = "quadratic"       #limb darkening model

        # Define the times at which to evaluate the fake transit
        t=normalized_corrected.time.value

        # Create the batman transit model
        m = batman.TransitModel(params, t)

        # Generate the fake light curve transit
        injected_model = m.light_curve(params)


        # Inject the fake transit into the real data
        injected_flux = normalized_corrected.flux.value + injected_model - 1.0


        lc_injected=normalized_corrected.copy()
        lc_injected.flux = injected_flux
        #fig,axs=plt.subplots(3,1,figsize=(10,10))
        planetrad = depth*stellar_radius * 9.73116 #convert the solar radii to jupiter radii
        #lc_injected.scatter(ax=axs[0],s=25,color='r',label='injected transit signal')
        #normalized_corrected.scatter(ax=axs[0],s=25)

        period_grid = np.linspace(0.4, 18, 1000)
        bls = lc_injected.to_periodogram(method='bls', period=period_grid, frequency_factor=500);
        #bls.plot(ax=axs[1],label=f'best p = {bls.period_at_max_power:.2f}');
        planet_b_period = bls.period_at_max_power
        planet_b_t0 = bls.transit_time_at_max_power
        planet_b_dur = bls.duration_at_max_power
        #lc_injected.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axs[2],label='')

        blsorig = normalized_corrected.to_periodogram(method='bls', period=period_grid, frequency_factor=500);
        origplanet_b_period = blsorig.period_at_max_power
        detectorig = abs((origplanet_b_period.value/period)-round(origplanet_b_period.value/period))

        detect = abs((planet_b_period.value/period)-round(planet_b_period.value/period))

        #axs[0].set_title(f'Rplanet(Rjup Radii) = {planetrad:.4f}, Midtime = {midtime:.2f}, Period = {period:.2f},operiod = {origplanet_b_period:.2f}')


        #print(f'detectval={detect}')
        accept_thresh=0.05
        if detectorig < accept_thresh:
            found = 0 #"False"

        elif detect < accept_thresh:
            found = 1 #"True"
        else:
            found = 0 #"False"

        #print(f'depth: {planetrad:.4f}              actual:{period:.4f}             calc:{planet_b_period:.4f}    Recover?:{found}')
        Rplanet.append(planetrad)
        Pinject.append(period)
        Pdetermine.append(planet_b_period.value)
        recover.append(found)

    output_table['Rplanet'] = Rplanet
    output_table['Pinject'] = Pinject
    output_table['Pdetermine'] = Pdetermine
    output_table['recover?'] = recover
        
    return output_table




output_table = output_table

radlist1=np.linspace(0.2,1.9,18)
radlist2=np.linspace(0.3,2,18)

detectionmatrix = np.empty((1,18))

for r1,r2 in zip(radlist1,radlist2):
    p1 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(0, 1))]
    recoverpercentp1 = (p1['recover?'].sum())/len(p1.index)
    #print(recoverpercentp1)

    p2 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(1, 2))]
    recoverpercentp2 = (p2['recover?'].sum())/len(p2.index)
    #print(recoverpercentp2)

    p3 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(2, 3))]
    recoverpercentp3 = (p3['recover?'].sum())/len(p3.index)
    #print(recoverpercentp3)

    p4 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(3, 4))]
    recoverpercentp4 = (p4['recover?'].sum())/len(p4.index)
    #print(recoverpercentp4)

    p5 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(4, 5))]
    recoverpercentp5 = (p5['recover?'].sum())/len(p5.index)
    #print(recoverpercentp5)

    p6 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(5, 6))]
    recoverpercentp6 = (p6['recover?'].sum())/len(p6.index)
    #print(recoverpercentp6)

    p7 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(6, 7))]
    recoverpercentp7 = (p7['recover?'].sum())/len(p7.index)
    #print(recoverpercentp7)

    p8 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(7, 8))]
    recoverpercentp8 = (p8['recover?'].sum())/len(p8.index)
    #print(recoverpercentp8)

    p9 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(8, 9))]
    recoverpercentp9 = (p9['recover?'].sum())/len(p9.index)
    #print(recoverpercentp9)
    #print('           ')

    
    p10 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(9, 10))]
    recoverpercentp10 = (p10['recover?'].sum())/len(p10.index)
    #print(recoverpercentp1)

    p11 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(10, 11))]
    recoverpercentp11 = (p11['recover?'].sum())/len(p11.index)
    #print(recoverpercentp2)

    p12 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(11, 12))]
    recoverpercentp12 = (p12['recover?'].sum())/len(p12.index)
    #print(recoverpercentp3)

    p13 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(12, 13))]
    recoverpercentp13 = (p13['recover?'].sum())/len(p13.index)
    #print(recoverpercentp4)

    p14 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(13, 14))]
    recoverpercentp14 = (p14['recover?'].sum())/len(p14.index)
    #print(recoverpercentp5)

    p15 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(14, 15))]
    recoverpercentp15 = (p15['recover?'].sum())/len(p15.index)
    #print(recoverpercentp6)

    p16 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(15, 16))]
    recoverpercentp16 = (p16['recover?'].sum())/len(p16.index)
    #print(recoverpercentp7)

    p17 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(16, 17))]
    recoverpercentp17 = (p17['recover?'].sum())/len(p17.index)
    #print(recoverpercentp8)

    p18 = output_table[(output_table['Rplanet'].between(r1, r2)) & (output_table['Pinject'].between(17, 18))]
    recoverpercentp18 = (p18['recover?'].sum())/len(p18.index)    
    
    matrixgrid = np.array([[recoverpercentp1,recoverpercentp2,recoverpercentp3,recoverpercentp4,recoverpercentp5,
                            recoverpercentp6,recoverpercentp7,recoverpercentp8,recoverpercentp9,recoverpercentp10,
                            recoverpercentp11,recoverpercentp12,recoverpercentp13,recoverpercentp14,
                            recoverpercentp15,recoverpercentp16,recoverpercentp17,recoverpercentp18]])
    detectionmatrix = np.concatenate((detectionmatrix, matrixgrid), axis=0)

detectionmatrix = np.delete(detectionmatrix, obj=0, axis=0)



fig, ax = plt.subplots(figsize=(10, 10))
# Using matshow here just because it sets the ticks up nicely. imshow is faster.
ax.matshow(detectionmatrix, cmap='plasma')

for (i, j), z in np.ndenumerate(detectionmatrix):
    ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    ax.margins(x=0)

ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5])
ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])

ax.set_yticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5])
ax.set_yticklabels([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2])

plt.gca().invert_yaxis()

plt.tick_params(labelbottom=True, labeltop=False)
plt.title(f'INSERT TITLE HERE')
plt.xlabel('Period [days]')
plt.ylabel('R_planet [R_Jup]')

#plt.xscale('log')

#ax.scatter(output_table['Pinject'],output_table['Rplanet'])

plt.show() 
