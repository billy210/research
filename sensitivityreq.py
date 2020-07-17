import numpy as np
from astropy import units as u
from astropy import constants as const

nu = (230 * u.GHz).cgs
Knu= 0.023 * u.cm**2/u.g #230 GHZ (Lee, Williams, Creza 2011)

dist = (131* u.pc).cgs

Rdisk = ([75,100,150]* u.au).cgs
Tdisk = (20* u.K).cgs
Mdisk = (0.009*u.solMass).cgs #MMEN

Ncol = Mdisk/(np.pi*(Rdisk)**2)

tau = Knu * Ncol

BBmod = (2*(const.h).cgs*nu**3)/((const.c).cgs**2)* (np.e**(((const.h).cgs*nu)/((const.k_B).cgs*Tdisk))-1)**-1 * (1-np.e**-tau)

Snu = (BBmod * (Rdisk**2/(4*dist**2))*4*np.pi).to(u.mJy)

print(Snu)
'''
###signal to noise time calc

SNR = 10      #desired snr
noise = 0.43* u.mJy #calculated from SMA calculator, mJy
t0 = 6.52*u.hr    #time in hrs for an SMA track

time= t0*(SNR/(Snu/noise))**2

print(f'estimated on source time for SNR of {SNR} = {time}')
'''
