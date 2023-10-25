#######################################################################################
#version with tess period removal has unknown bugs, proceeding with this for now#######
#######################################################################################
def injectplanettest(arg):
    idx,row = arg
    #binned_lc.scatter()
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

    # Define the times at which to evaluate the fake transit
    t=binned_lc.time.value

    # Create the batman transit model
    m = batman.TransitModel(params, t)

    # Generate the fake light curve transit
    injected_model = m.light_curve(params)


    # Inject the fake transit into the real data
    injected_flux = binned_lc.flux.value + injected_model - 1.0


    lc_injected=binned_lc.copy()
    lc_injected.flux = injected_flux
    #fig,axs=plt.subplots(3,1,figsize=(10,10))
    planetrad = depth*stellar_radius * 9.73116 #convert the solar radii to jupiter radii
    #lc_injected.scatter(ax=axs[0],s=25,color='r',label='injected transit signal')
    #normalized_corrected.scatter(ax=axs[0],s=25)
    
    period_grid = np.linspace(0.4, 18, 10000)
    bls = lc_injected.to_periodogram(method='bls', period=period_grid, frequency_factor=700);
    #bls.plot(ax=axs[1],label=f'best p = {bls.period_at_max_power:.2f}');
    planet_b_period = bls.period_at_max_power
    planet_b_t0 = bls.transit_time_at_max_power
    planet_b_dur = bls.duration_at_max_power
    #lc_injected.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter(ax=axs[2],label='')

    blsorig = binned_lc.to_periodogram(method='bls', period=period_grid, frequency_factor=700);
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






def vary_paramsmult(func, input_lc, df, stellar_radius=1.325, trials=10000, num_processes=1):
    t0 = time.time()
    binned_lc = input_lc
    def init():
        global binned_lc
        binned_lc = input_lc
    
    with mp.Pool(num_processes,initializer=init,maxtasksperchild=10) as pool:
        binned_lc.scatter()
        #print(output_table['depth'])
        rad_min = 0.02115/stellar_radius #now in R_hoststar
        rad_max = 0.2/stellar_radius #now in R_hoststar

        depths = np.random.uniform(rad_min, rad_max, trials)  # random transit depths to inject
        midtimes = np.random.uniform(min(binned_lc.time.value), max(binned_lc.time.value), trials)  # mid-transit times to inject if you want
        periods = np.random.uniform(0.4,18,trials) # periods to inject
        
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

#output_table_currents3 = pd.DataFrame()
#output_table_currents3 = vary_paramsmult(injectplanettest,s3lcbin, output_table_currents3, stellar_radius=0.681, trials=1, num_processes=128)
