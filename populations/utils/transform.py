"""
Parameter transforms
"""
import numpy as np
from scipy.stats import powerlaw

def _to_mchirp(samples):
    m1, m2 = samples["m1"], samples["m2"]
    return (m1 * m2)**(3./5) / (m1 + m2)**(1./5)

def _to_q(samples):
    m1, m2 = samples["m1"], samples["m2"]
    q1 = m1/m2
    q2 = m2/m1
    return np.minimum(q1,q2)

def _to_eta(samples):
    m1, m2 = samples["m1"], samples["m2"]
    return (m1 * m2) / (m1 + m2)**2

def _to_tilt1(samples):
    if "theta1" not in samples.keys():
        return np.arccos(samples["cos_t1"])
    else:
        return samples["theta1"]

def _to_tilt2(samples):
    if "theta2" not in samples.keys():
        return np.arccos(samples["cos_t2"])
    else:
        return samples["theta2"]

def _to_chieff(samples):
    m1, m2 = samples["m1"], samples["m2"]
    a1, a2 = samples["a1"], samples["a2"]
    t1, t2 = samples["theta1"], samples["theta2"]
    return (m1*a1*np.cos(t1) + m2*a2*np.cos(t2))/(m1+m2)

def _uniform_spinmag(samples):
    a1 = np.random.uniform(0,1, len(samples))
    a2 = np.random.uniform(0,1, len(samples))
    return a1, a2

def _isotropic_spinmag(samples):
    a1 = powerlaw.rvs(3, size=len(samples))
    a2 = powerlaw.rvs(3, size=len(samples))
    return a1, a2

_DEFAULT_TRANSFORMS = {
    "mchirp": _to_mchirp,
    "q": _to_q, 
    "eta": _to_eta,
    "theta1": _to_tilt1,
    "theta2": _to_tilt2
}
