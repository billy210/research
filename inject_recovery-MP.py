#Same as the previous injection recovery program, but now involves multiprocessing. While batman has its own multithread, that wasnt our setback.
# By updating this code and running it on 32 cores, it took the runtime for 5000 trials from 10.5 mins to just under 0.71 mins. Substaintial as we move forward. 

def injectplanet(arg):
    idx,row = arg
    depth = row['depths']
    midtime = row['midtimes']
    period=  row['periods']
    
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = row['midtimes']                   #time of inferior conjunction
    params.per =  row['periods']                 #orbital period
    params.rp = row['depths']                     #planet radius (in units of stellar radii)
    stellar_radius = row['stellar_radius']
    normalized_corrected = lcbin
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
    bls = lc_injected.to_periodogram(method='bls', period=period_grid, frequency_factor=600);
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
    if (detectorig < accept_thresh) & (bls.max_power <= blsorig.max_power):
        found = 0 #"False"
    elif (detectorig < accept_thresh) & (bls.max_power > blsorig.max_power):
        found = 1 #"True"

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






def vary_paramsmult(func, normalized_corrected, df, stellar_radius=1.325, trials=1000, num_processes=1):
    
    t0 = time.time()
    with mp.Pool(num_processes,maxtasksperchild=10) as pool:
        #print(output_table['depth'])
        rad_min = 0.02115/stellar_radius #now in R_hoststar
        rad_max = 0.2/stellar_radius #now in R_hoststar

        depths = np.random.uniform(rad_min, rad_max, trials)  # random transit depths to inject
        midtimes = np.random.uniform(min(normalized_corrected.time.value), max(normalized_corrected.time.value), trials)  # mid-transit times to inject if you want
        periods = np.random.uniform(0.3,18,trials) # periods to inject
        
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



output_table2 = pd.DataFrame()
output_table2 = vary_paramsmult(injectplanet,lcbin, output_table2, stellar_radius=0.994, trials=5000, num_processes=32)