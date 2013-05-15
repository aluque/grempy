""" Code to solve the Langevin equation simultaneously with the FDTD. """

from numpy import *
import scipy.constants as co
from scipy.interpolate import interp1d, splrep, splev

from fdtd import (CPML, Cylindrical, CylindricalCPML, staggered,
                  Box, Dimensions, CylBox, CylDimensions, avg)

X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2
TOWNSEND = 1e-21

class CylindricalLangevin(Cylindrical):
    """ A simulation to solve the Langevin equation coupled with the
    electron density evolution equation. """
    def __init__(self, box, dim):
        """ Initializes a Langevin simulator.
        * mun is the electron mobility times ngas (a scalar).
        * ngas is the density of neutrals.  Its shape must be ngas[nr, nz]
          but the r dimension can be broadcasted away (i.e. the shape
          can be ngas[1, nz] and we assume that the density does not depend on
          r.
        * ne is the initial electron density.  Again, the r dimension
          may be broadcasted away, but in that case the array is expanded
          during the initialization.
        """

        super(CylindricalLangevin, self).__init__(box, dim)

        self.ngas = empty((1, dim.nz + 1))
        self.ne = empty((dim.nr + 1, dim.nz + 1))

        # J is located along integer coordinates (all 2 components at the
        # same places).  Hence we do not need to use the staggered class
        self.j = zeros((dim.nr + 1, dim.nz + 1, 3))
        self.e = zeros((dim.nr + 1, dim.nz + 1, 3))
        self.eabs = zeros((dim.nr + 1, dim.nz + 1))
        self.dens_update_lower = 0.0

    def load_ne(self, fname):
        """ Loads the electron density profile and interpolates it into z. """
        iri = loadtxt(fname)
        h, n = iri[:, 0] * co.kilo, iri[:, 1] * co.centi**-3

        #ipol = interp1d(h, log(n), bounds_error=False, fill_value=-inf)
        #n2 = exp(ipol(z))

        tck = splrep(h, log(n), k=1)
        self.ne[:, :] = exp(splev(self.zf, tck))[newaxis, :]


    def load_ngas(self, fname):
        """ Loads the density of neutrals and interpolates into z. """
        atm = loadtxt(fname)
        h = atm[:, 0] * co.kilo
        n = atm[:, 1] * co.centi**-3

        ipol = interp1d(h, log(n), bounds_error=False, fill_value=-inf)
        self.ngas[:, :] = exp(ipol(self.zf))[newaxis, :]
        self._update_mu()


    def _update_mu(self):
        try:
            self.mu = self.mun / self.ngas
            self.nu = co.elementary_charge / (self.mu * co.electron_mass)
        except AttributeError:
            pass

    def set_mun(self, mun):
        self.mun = mun
        self._update_mu()


    def load_ionization(self, fname):
        en, ionization = loadtxt(fname, unpack=True)
        en = r_[0.0, en]
        ionization = r_[0.0, ionization]

        self.ionization_k = interp1d(en * TOWNSEND, ionization)


    def set_dt(self, dt, **kwargs):
        super(CylindricalLangevin, self).set_dt(dt, **kwargs)
        
        # We set a cutoff to avoid overflows.
        # Following Lee 1999, we calculate now exp(-nu t) * exp(nu t)
        # which is 1 but only as long as the two exponentials can be
        # calculated.
        expnu = exp(where(self.nu * dt < 100, self.nu * dt, 100))

        exp_nu = 1. / expnu
        
        self.A = exp_nu
        # Kp is K without the wp2 factor, which is the part that depends on
        # the electron dens.
        self.Kp = co.epsilon_0 * (1. - self.A) / self.nu

        self.A[:] = where(isfinite(self.A), self.A, 0)
        self.Kp[:] = where(isfinite(self.Kp), self.Kp, 0)
        self.K = empty_like(self.ne)
       

    def interpolate_e(self):
        """ Interpolate E{x, y, z} at the locations of J. """
        
        self.e[:, :, R] = 0.5 * self.er.avg(R)
        self.e[:, :, Z_] = 0.5 * self.ez.avg(Z_)
        self.e[:, :, PHI] = 0.5 * self.ephi.v
        
        self.eabs[:, :] = sqrt(sum(self.e**2, axis=2))


    def update_j(self):
        """ Updates j from time-step n-1/2 to n+1/2. """
        wp = sqrt(co.elementary_charge**2 
                  * self.ne / co.electron_mass / co.epsilon_0)
        wp2 = wp**2
        self.K[:] = self.Kp * wp2
        self.j[:] = (self.A[:, :, newaxis] * self.j
                     + self.K[:, :, newaxis] * self.e[:, :, :])


    def j_(self, coord):
        """ Returns the coord component of j. """
        slices = [s_[:]] * rank(self.j)
        slices[-1] = coord

        return self.j[slices]

    
    def update_e(self):
        super(CylindricalLangevin, self).update_e()
    
        self.er.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * avg(self.j_(R), axis=R))
        self.ez.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * avg(self.j_(Z_), axis=Z_))
        self.ephi.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * self.j_(PHI))

        self.interpolate_e()
        self.update_ne()


    def update_h(self):
        # We plug update_j into update_h of the general simulation.
        
        super(CylindricalLangevin, self).update_h()
        self.update_j()
        

    def update_ne(self):
        """ Updates the electron density.  Uses an implicit method,
        so we need to call this after update_e. """
        nu = self.ngas * self.ionization_k(self.eabs / self.ngas)
        
        nu[:, :] = where(self.zf[newaxis, :] >= self.dens_update_lower,
                         nu[:, :], 0.0)
        self.ne[:, :] /= (1 - self.dt * nu)


    def add_cpml_boundaries(self, n, filter=None):
        super(CylindricalLangevin, self).add_cpml_boundaries(n, filter=filter)
        self.nu[:, -n:] = 0


    def save_global(self, g):
        super(CylindricalLangevin, self).save_global(g)

        g.create_dataset('ngas', data=self.ngas, compression='gzip')


    def save(self, g):
        super(CylindricalLangevin, self).save(g)
        g.create_dataset('j', data=self.j, compression='gzip')
        g.create_dataset('ne', data=self.ne, compression='gzip')


    def load_data(self, g):
        super(CylindricalLangevin, self).load_data(g)

        self.mu = self.mun / self.ngas
        self.nu = co.elementary_charge / (self.mu * co.electron_mass)

        self.j[:] = array(g['j'])
        self.ne[:] = array(g['ne'])
        self.interpolate_e()

        
    @staticmethod
    def load(g, step, set_dt=False, c=None):
        """ Loads an instance from g and initializes all its data. """
        if isinstance(g, str):
            import h5py
            g = h5py.File(g, 'r')
            
        box = CylBox(*g['box'])
        dim = CylDimensions(*g['dim'])

        if c is None:
            c = CylindricalLangevin

        instance = c(box, dim)
        instance.ngas[:, :] = array(g['ngas'])
        
        instance.set_mun(g.attrs['mu_N'])
        
        gstep = g['steps/' + step]

        for key, val in gstep.iteritems():
            if key[:5] == 'cpml_':
                cpml = CylindricalCPML.load(val, instance)
                instance.cpml.append(cpml)
                
        instance.load_data(gstep)

        if set_dt:
            instance.set_dt(self.dt, init=False)
        else:
            instance.dt = g.attrs['dt']

        instance.te = gstep.attrs['te']
        instance.th = gstep.attrs['th']

        return instance


if __name__ == '__main__':
    main()
