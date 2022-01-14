#for calculating disk mass estimates:
from astropy.modeling.models import BlackBody
from astropy import units as u
def OptThinMass(S,d=1*u.kpc,wav=1*u.mm,kappa=0.0114*u.cm**2/u.g,T=20*u.K):
    bb = BlackBody(temperature=T)      
    solar =  (S*d**2/kappa/bb(wav)/u.sr).to(u.M_sun)
    earthly =  (S*d**2/kappa/bb(wav)/u.sr).to(u.M_earth)
    return solar,earthly #returns mass in solar masses and earth masses
