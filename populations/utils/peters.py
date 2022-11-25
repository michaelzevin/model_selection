from __future__ import division
from scipy import integrate
from scipy.integrate import ode
import numpy as np

def deda_peters(a,e):
	num = 12*a*(1+(73./24)*e**2 + (37./96)*e**4)
	denom = 19*e*(1-e**2)*(1+(121./304)*e**2)
	return denom/num

def inspiral_time_peters(a0,e0,m1,m2,af=0):
	"""
	Computes the inspiral time, in Gyr, for a binary
	a0 in Au, and masses in solar masses
	
	if different af is given, computes the time from a0,e0
	to that final semi-major axis
	
	for af=0, just returns inspiral time
	for af!=0, returns (t_insp,af,ef)
	"""
	coef = 6.086768e-11 #G^3 / c^5 in au, gigayear, solar mass units
	beta = (64./5.) * coef * m1 * m2 * (m1+m2)
	
	if e0 == 0:
		if not af == 0:
			print("ERROR: doesn't work for circular binaries")
			return 0
		return a0**4 / (4*beta)
	
	c0 = a0 * (1.-e0**2.) * e0**(-12./19.) * (1.+(121./304.)*e0**2.)**(-870./2299.)
	
	if af == 0:
		eFinal = 0.
	else:
		r = ode(deda_peters)
		r.set_integrator('lsoda')
		r.set_initial_value(e0,a0)
		r.integrate(af)
		if not r.successful():
			print("ERROR, Integrator failed!")
		else:
			eFinal = r.y[0]	  
	
	time_integrand = lambda e: e**(29./19.)*(1.+(121./304.)*e**2.)**(1181./2299.) / (1.-e**2.)**1.5
	integral,abserr = integrate.quad(time_integrand,eFinal,e0)
	
	if af==0:
		return integral * (12./19.) * c0**4. / beta
	else:
		return (integral * (12./19.) * c0**4. / beta,af,eFinal)

def a_at_fLow(m1,m2,fLow = 5):
	"""
	Computes the semi-major axis at an orbital frequency of fLow
	Masses in solar masses, fLow in Hz
	"""
	G = 3.9652611e-14 # G in au,solar mass, seconds
	quant = G*(m1+m2) / (4*np.pi**2 * fLow**2)
	return quant**(1./3)

def eccentricity_at_a(m1,m2,a_0,e_0,a):
	"""
	Computes the eccentricity at a given semi-major axis a 
	
	Masses are in solar masses, a_0 in AU
	
	"""
	r = ode(deda_peters)
	r.set_integrator('lsoda')
	r.set_initial_value(e_0,a_0)
	
	r.integrate(a)
	
	if not r.successful():
		print("ERROR, Integrator failed!")
	else:
		return r.y[0]

def eccentricity_at_fLow(m1,m2,a_0,e_0,fLow=10):
	"""
	Computes the eccentricity at a given fLow, assuming f_gw = 2*f_orb
	
	Masses are in solar masses, a_0 in AU
	
	NOTE!!! The frequency here is the gravitational-wave frequency.  
	"""
	a_low = a_at_fLow(m1,m2,fLow/2.)

	return eccentricity_at_a(m1,m2,a_0,e_0,a_low)

	
def au_to_period(a,m):
    """
    Returns the period (in days) for a binary
    with mass m (solar masses) and semi-major axis a (AU)
    """
    g = 2.96e-4 # G in days,AU,solar masses
    return np.sqrt(a**3 * 4*np.pi**2 / (g*m))

def eccentric_gwave_freq(a,m,e):
	"""
	returns the gravitational-wave frequency for a binary at seperation a (AU), mass
	M (solar masses), and eccentricity e, using the formula from Wen 2003
	"""


	freq = 1 / (86400*au_to_period(a,m))
	return 2*freq*pow(1+e,1.1954)/pow(1-e*e,1.5)

def eccentricity_at_eccentric_fLow(m1,m2,a_0,e_0,fLow=10,retHigh = False):
	"""
	Computes the eccentricity at a given fLow using the peak frequency from Wen
	2003
	
	Masses are in solar masses, a_0 in AU.  The frequency here is the
	gravitational-wave frequency.

	Note that it is possible that, for binaries that merge in fewbody, there is
	no solution, since the binary was pushed into the LIGO band above 10Hz.  In
	this case, there is no a/e from that reference that will give you 10Hz, so
	this just returns e_0 by default, and 0.99 if retHigh is true
	"""
	from scipy.optimize import brentq

	ecc_at_a = lambda a: eccentricity_at_a(m1,m2,a_0,e_0,a)
	freq_at_a = lambda a: eccentric_gwave_freq(a,m1+m2,ecc_at_a(a))
	zero_eq = lambda a: freq_at_a(a) - fLow

	lower_start = zero_eq(1e-10)
	upper_start = zero_eq(1)

	if (np.sign(lower_start) == np.sign(upper_start) or 
		np.isnan(lower_start) or np.isnan(upper_start)):
		if retHigh:
			return 0.99999
		else:
			return e_0
	else:	
		a_low = brentq(zero_eq,1e-10,1)
		return ecc_at_a(a_low) 

def timescale_perihelion_precession(a,e,M):
	"""
	Computes and returns the timescale for a single revolution of the periapse
	of a binary

	Time is in seconds, M in solar masses, a in AU
	"""
	G = 3.9652611e-14 # G in au,solar mass, seconds
	c = 0.0020039888 # C in au/sec

	return c**2 * a**2.5 * (1 - e**2) / (3*(G*M)**1.5)

def timescale_spinorbit_precession(a,e,M,chi_eff):
	"""
	Computes and returns the timescale for a single revolution of L about J for
	a spinning binary (the Lense-Thirring contribution)

	Time is in seconds, M in solar masses, a in AU. chi_eff is [-1,1]
	"""
	G = 3.9652611e-14 # G in au,solar mass, seconds
	c = 0.0020039888 # C in au/sec

	return c**3 * a**3 * (1 - e**2)**1.5 / (2*chi_eff*(G*M)**2)

def calc_c0(a0,e0):
    num = a0*(1-e0**2)
    denom = e0**(12./19) * (1+(121./304)*e0**2)**(870./2299)
    return num/denom

def a_at_eccentricity(a0,e0,e):
    """
    Computes semi-major axis at a specified eccentricity, given 
    starting orbital conditions a0 and e0

    a in AU
    """
    c0 = calc_c0(a0,e0)

    num = c0*e**(12./19) * (1 + (121./304)*e**2)**(870./2299)
    denom = (1-e**2)
    return num/denom

def Rperi_at_eccentricity(a0,e0,e):
    """
    Computes periapse distance at a specified eccentricity, given 
    starting orbital conditions a0 and e0

    a in AU
    """
    c0 = calc_c0(a0,e0)

    num = c0*e**(12./19) * (1 + (121./304)*e**2)**(870./2299) * (1-e)
    denom = (1-e**2)
    return num/denom

def Rapo_at_eccentricity(a0,e0,e):
    """
    Computes apoapse distance at a specified eccentricity, given 
    starting orbital conditions a0 and e0

    a in AU
    """
    c0 = calc_c0(a0,e0)

    num = c0*e**(12./19) * (1 + (121./304)*e**2)**(870./2299) * (1+e)
    denom = (1-e**2)
    return num/denom
