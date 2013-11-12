import os.path

from numpy import *
import scipy.constants as co
from scipy.interpolate import splrep, splev

from plugin import Plugin

X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2

Td = 1e-17 * co.centi**2


class Radiative(Plugin):
    PHOTON_NAMES = ["fulcher", "cont"]
    N_PHOTON_TYPES = len(PHOTON_NAMES)

    def __init__(self, params, load_files=True):
        if load_files:
            fname = os.path.join(params.input_dir,
                                 'rates.dat')

            self.load_rates(fname)


    def load_rates(self, rates_file):
        self.rates = [
            asfunction(rates_file,
                       xfactor=Td,
                       usecols=(1, 14),
                       log=False),
            asfunction(rates_file,
                       xfactor=Td,
                       usecols=(1, 12),
                       log=False)]
        
    def initialize(self, sim):
        # Every species is located at the nodes, i.e. the same locations
        # as the interpolated electric field.
        sim.nphotons = zeros((sim.dim.nr + 1, sim.dim.nz + 1, 
                              self.N_PHOTON_TYPES))


    def update_h(self, sim):
        # First, calculate the E/N
        en = sqrt(sum(sim.e**2, axis=2)) / sim.ngas
        en = where(isfinite(en), en, 0)
        
        # Then update each photon species
        for i in xrange(self.N_PHOTON_TYPES):
            # We have to unravel and reshape the array bc splev only supports
            # 1d arrays.
            k = self.rates[i](ravel(en)).reshape(en.shape)
            
            sim.nphotons[:, :, i] += sim.dt * k * sim.ne * sim.ngas


    def save(self, sim, g):
        for i in xrange(self.N_PHOTON_TYPES):
            g.create_dataset(self.PHOTON_NAMES[i],
                             data=sim.nphotons[:, :, i], 
                             compression='gzip')


    def load_data(self, sim, g):
        for i in xrange(self.N_PHOTON_TYPES):
            sim.nphotons[:, :, i] = array(g[self.PHOTON_NAMES[i]])


# Utility functions
def asfunction(f, xfactor=1, yfactor=1, log=False, **kwargs):
    """ Returns a function from f.  f can be:
    - A function: then we return f
    - A tuple (x, y): then we return interp1d(x, y,...)
    - A filename: then we read x, y from the filename and return
      asfunction(x, y).

    """
    if isinstance(f, tuple):
        x, y = f
        constructor = Spline if not log else LogSpline
        #return constructor(x * xfactor, y * yfactor,
        #                   bounds_error=False, fill_value=0)
        return constructor(x * xfactor, y * yfactor, k=1)
        
    elif isinstance(f, str) or isinstance(f, unicode):
        tpl = tuple(loadtxt(f, unpack=True, **kwargs))
        return asfunction(tpl, xfactor=xfactor, yfactor=yfactor, log=log)
    
    if yfactor == 1 and xfactor == 1:
        return f

    else:
        return lambda(x): yfactor * f(x / xfactor)


class LogSpline(object):
    def __init__(self, x, y, **kwargs):
        self.tck = splrep(x, log(y), **kwargs)

    def __call__(self, x):
        return exp(splev(x, self.tck))


class Spline(object):
    def __init__(self, x, y, **kwargs):
        self.tck = splrep(x, y, **kwargs)

    def __call__(self, x):
        return splev(x, self.tck)
        
