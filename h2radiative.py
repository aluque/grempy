from numpy import *
import scipy.constants as co
from scipy.interpolate import splrep, splev

from langevin import CylindricalLangevin

X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2

Td = 1e-17 * co.centi**2


class Radiative2d(CylindricalLangevin):
    PHOTON_NAMES = ["fulcher", "cont"]
    
    def __init__(self, *args, **kwargs):
        super(Radiative2d, self).__init__(*args, **kwargs)

        self.nphotons = 2
        
        # Every species is located at the nodes, i.e. the same locations
        # as the interpolated electric field.
        self.n = zeros((self.dim.nr + 1, self.dim.nz + 1, self.nphotons))

        # These are set with set_densities
        self.nt = None
        self.ne = None

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


    def step_photons(self):
        # First, calculate the E/N
        en = sqrt(sum(self.e**2, axis=2)) / self.nt
        en = where(isfinite(en), en, 0)
        
        # Then update each photon species
        for i in xrange(self.nphotons):
            # We have to unravel and reshape the array bc splev only supports
            # 1d arrays.
            k = self.rates[i](ravel(en)).reshape(en.shape)
            
            self.n[:, :, i] += self.dt * k * self.ne * self.nt
            
        
    def set_densities(self, nt, ne):
        self.nt = nt
        self.ne = ne
        

    def update_h(self):
        # We plug update_j into update_h of the general simulation.
        
        super(Radiative2d, self).update_h()
        self.step_photons()


    def save(self, g):
        super(Radiative2d, self).save(g)

        for i in xrange(self.nphotons):
            g.create_dataset(self.PHOTON_NAMES[i],
                             data=self.n[:, :, i], 
                             compression='gzip')


    def load_data(self, g):
        super(Radiative2d, self).load_data(g)

        for i in xrange(self.nphotons):
            self.n[:, :, i] = array(g[self.PHOTON_NAMES[i]])
            

    @staticmethod
    def load(g, step, set_dt=False):
        instance = CylindricalLangevin.load(g, step, 
                                            set_dt=set_dt, c=Radiative2d)
    
        return instance
    


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
        
