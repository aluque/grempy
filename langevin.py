""" Code to solve the Langevin equation simultaneously with the FDTD. """

from numpy import *
import scipy.constants as co

from fdtd import (CPML, Cylindrical, CylindricalCPML, staggered,
                  Box, Dimensions, CylBox, CylDimensions, avg)

X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2


class CylindricalLangevin(Cylindrical):
    """ A simulation to solve the Langevin equation. """
    def __init__(self, box, dim, nspecies, nu, wp):
        """ Initializes a Langevin simulator.
        * nspecies is the number of species that will be followed.
        * nu is the effective collision freq. of each species.  Its shape
          must be nu[nspecies] or nu[nr, nz, nspecies]
        * wp is the plasma frequency, its shape has the same options as nu
        Note that in cylindrical symmetry we disregard any geo-magnetic field.
        """

        super(CylindricalLangevin, self).__init__(box, dim)

        self.nspecies = nspecies
        self.nu = nu
        self.wp = wp
        self.wp2 = wp**2

        # J is located along integer coordinates (all 2 components at the
        # same places).  Hence we do not need to use the staggered class
        self.j = zeros((dim.nr + 1, dim.nz + 1, nspecies, 3))
        self.e = zeros((dim.nr + 1, dim.nz + 1, 3))


    def set_dt(self, dt):
        super(CylindricalLangevin, self).set_dt(dt)

        # We set a cutoff to avoid overflows.
        # Following Lee 1999, we calculate now exp(-nu t) * exp(nu t)
        # which is 1 but only as long as the two exponentials can be
        # calculated.
        expnu = exp(where(self.nu * dt < 100, self.nu * dt, 100))

        exp_nu = 1. / expnu
        
        self.A = exp_nu
        self.K = co.epsilon_0 * self.wp2 * (1. - self.A) / self.nu

        self.A[:] = where(isfinite(self.A), self.A, 0)
        self.K[:] = where(isfinite(self.K), self.K, 0)


        
    def interpolate_e(self):
        """ Interpolate E{x, y, z} at the locations of J. """
        
        self.e[:, :, R] = 0.5 * self.er.avg(R)
        self.e[:, :, Z_] = 0.5 * self.ez.avg(Z_)
        self.e[:, :, PHI] = 0.5 * self.ephi.v
        
        
    def update_j(self):
        """ Updates j from time-step n-1/2 to n+1/2. """
        self.interpolate_e()

        self.j[:] = (self.A[:, :, :, newaxis] * self.j
                     + self.K[:, :, :, newaxis] * self.e[:, :, newaxis, :])


    def j_(self, coord):
        """ Returns the coord component of j. """
        slices = [s_[:]] * rank(self.j)
        slices[-1] = coord

        return self.j[slices]

    
    def update_e(self):
        super(CylindricalLangevin, self).update_e()
    
        # The sums here are over species.        
        self.er.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * sum(avg(self.j_(R), axis=R), axis=-1))
        self.ez.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * sum(avg(self.j_(Z_), axis=Z_), axis=-1))
        self.ephi.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * sum(self.j_(PHI), axis=-1))


    def update_h(self):
        # We plug update_j into update_h of the general simulation.
        
        super(CylindricalLangevin, self).update_h()
        self.update_j()

    def save_global(self, g):
        super(CylindricalLangevin, self).save_global(g)
        g.attrs['nspecies'] = self.nspecies

        g.create_dataset('nu', data=self.nu, compression='gzip')
        g.create_dataset('wp', data=self.wp, compression='gzip')
        

    def save(self, g):
        super(CylindricalLangevin, self).save(g)
        g.create_dataset('j', data=self.j, compression='gzip')


    def load_data(self, g):
        super(CylindricalLangevin, self).load_data(g)

        self.j[:] = array(g['j'])

        self.interpolate_e()

        
    @staticmethod
    def load(g, step, set_dt=False, c=None):
        """ Loads an instance from g and initializes all its data. """
        if isinstance(g, str):
            import h5py
            g = h5py.File(g, 'r')
            
        box = CylBox(*g['box'])
        dim = CylDimensions(*g['dim'])
        nspecies = g.attrs['nspecies']

        nu = array(g['nu'])
        wp = array(g['wp'])
        if c is None:
            c = CylindricalLangevin

        instance = c(box, dim, nspecies, nu, wp)

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
