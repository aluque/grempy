""" Code to solve the Langevin equation simultaneously with the FDTD. """

from numpy import *
import scipy.constants as co

from fdtd import (CPML, Cartesian3d, Cylindrical, CylindricalCPML, staggered,
                  Box, Dimensions, CylBox, CylDimensions, avg)

X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2

class Langevin(Cartesian3d):
    """ A simulation to solve the Langevin equation. """
    def __init__(self, box, dim, nspecies, nu, wp, wb):
        """ Initializes a Langevin simulator.
        * nspecies is the number of species that will be followed.
        * nu is the effective collision freq. of each species.  Its shape
          must be nu[nspecies] or nu[nx, ny, nz, nspecies]
        * wp is the plasma frequency, its shape has the same options as nu
        * wb is a vectorial magnetic gyrofrequency.  Its shape may be
          wb[nspecies, 3] or wb[nx, ny, nz, nspecies, 3].
        """

        super(Langevin, self).__init__(box, dim)

        self.nspecies = nspecies
        self.nu = nu
        self.wp = wp
        self.wp2 = wp**2
        self.wb = wb

        self.wb2 = sum(self.wb**2, axis=-1)
        self.wb_ = sqrt(self.wb2)

        # J is located along integer coordinates (all 3 components at the
        # same places).  Hence we do not need to use the staggered class
        self.j = zeros((dim.nx + 1, dim.ny + 1, dim.nz + 1, nspecies, 3))
        self.e = zeros((dim.nx + 1, dim.ny + 1, dim.nz + 1, 3))


    def set_dt(self, dt):
        super(Langevin, self).set_dt(dt)

        # We set a cutoff to avoid overflows.
        # Following Lee 1999, we calculate now exp(-nu t) * exp(nu t)
        # which is 1 but only as long as the two exponentials can be
        # calculated.
        expnu = exp(where(self.nu * dt < 100, self.nu * dt, 100))
        
        exp_nu = 1. / expnu
        coswbt = cos(self.wb_ * dt)
        
        S1 = where(self.wb_ > 0, sin(self.wb_ * dt) / self.wb_, dt)
        
        C1 = where(self.wb2 > 0, (1 - coswbt) / self.wb2, dt**2 / 2)
        C2 = (expnu - 1.) / self.nu - C1 * self.nu - S1
        C3 = (self.nu * (expnu - coswbt)
              + self.wb_ * sin(self.wb_ * dt))
        C4 = expnu - coswbt - S1 * self.nu

        # To build the A and K matrices we will first put the two indices
        # as the first
        # later we will rearrange to move them to be the last indices.
        A = zeros((3, 3) + C4.shape)
        K = zeros((3, 3) + C4.shape)

        wbx, wby, wbz = [self.wb.T[i].T for i in (0, 1, 2)]
        
        A[0, 0] = C1 * wbx*wbx + coswbt
        A[0, 1] = C1 * wbx*wby - S1 * wbz
        A[0, 2] = C1 * wbx*wbz + S1 * wby

        A[1, 0] = C1 * wby*wbx + S1 * wbz
        A[1, 1] = C1 * wby*wby + coswbt
        A[1, 2] = C1 * wby*wbz - S1 * wbx

        A[2, 0] = C1 * wbz*wbx - S1 * wby
        A[2, 1] = C1 * wbz*wby + S1 * wbx
        A[2, 2] = C1 * wbz*wbz + coswbt

        K[0, 0] = C2 * wbx*wbx + C3
        K[0, 1] = C2 * wbx*wby - C4 * wbz
        K[0, 2] = C2 * wbx*wbz + C4 * wby

        K[1, 0] = C2 * wby*wbx + C4 * wbz
        K[1, 1] = C2 * wby*wby + C3
        K[1, 2] = C2 * wby*wbz - C4 * wbx

        K[2, 0] = C2 * wbz*wbx - C4 * wby
        K[2, 1] = C2 * wbz*wby + C4 * wbx
        K[2, 2] = C2 * wbz*wbz + C3

        trans = range(rank(A))
        trans = trans[2:] + [0, 1]

        self.A = transpose(exp_nu * A, trans)
        self.K = transpose((exp_nu / (self.wb2 + self.nu**2)) * K, trans)

        self.A[:] = where(isfinite(self.A), self.A, 0)
        self.K[:] = where(isfinite(self.K), self.K, 0)

        
    def interpolate_e(self):
        """ Interpolate E{x, y, z} at the locations of J. """
        
        self.e[:, :, :, X] = 0.5 * self.ex.avg(X)
        self.e[:, :, :, Y] = 0.5 * self.ey.avg(Y)
        self.e[:, :, :, Z] = 0.5 * self.ez.avg(Z)
        
        
    def update_j(self):
        """ Updates j from time-step n-1/2 to n+1/2. """
        self.interpolate_e()

        s_j = [s_[:]] * (rank(self.j) - 1) + [newaxis, s_[:]]
        s_e = [s_[:]] * (rank(self.e) - 1) + [newaxis, newaxis, s_[:]]
        s_wp2 = [s_[:]] * rank(self.wp2) + [newaxis,]

        t1 = sum(self.A * self.j[s_j], axis=-1)        
        t2 = co.epsilon_0 * self.wp2[s_wp2] * sum(self.K * self.e[s_e], axis=-1)
        
        # s_e = [s_[:]] * (rank(self.e) - 1) + [newaxis, s_[:]]
        # t2 = co.epsilon_0 * (self.wp2 / self.nu)[s_wp2] * self.e[s_e] 
        
        self.j[:] = t1 + t2


    def j_(self, coord):
        """ Returns the coord component of j. """
        slices = [s_[:]] * rank(self.j)
        slices[-1] = coord
        return self.j[slices]

    
    def update_e(self):
        super(Langevin, self).update_e()
    
        # The sums here are over species.        
        self.ex.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * sum(avg(self.j_(X), axis=X), axis=-1))
        self.ey.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * sum(avg(self.j_(Y), axis=Y), axis=-1))
        self.ez.v[:] -= (0.5 * self.dt / co.epsilon_0
                         * sum(avg(self.j_(Z), axis=Z), axis=-1))


    def update_h(self):
        # We plug update_j into update_h of the general simulation.
        
        super(Langevin, self).update_h()
        self.update_j()



    def save(self, g):
        super(Langevin, self).save(g)
        g.attrs['nspecies'] = self.nspecies

        g.create_dataset('nu', data=self.nu, compression='gzip')
        g.create_dataset('wp', data=self.wp, compression='gzip')
        g.create_dataset('wb', data=self.wb, compression='gzip')

        g.create_dataset('j', data=self.j, compression='gzip')


    def load_data(self, g):
        super(Langevin, self).load_data(g)

        self.nu[:] = array(g['nu'])
        self.wp[:] = array(g['wp'])
        self.wb[:] = array(g['wb'])
        self.j[:] = array(g['j'])

        self.interpolate_e()
        
    @staticmethod
    def load(g, set_dt=False):
        """ Loads an instance from g and initializes all its data. """
        box = Box(*g['box'])
        dim = Dimensions(*g['dim'])
        nspecies = g.attrs['nspecies']

        nu = array(g['nu'])
        wp = array(g['wp'])
        wb = array(g['wb'])
        
        instance = Langevin(box, dim, nspecies, nu, wp, wb)
        for key, val in g.iteritems():
            if key[:5] == 'cpml_':
                cpml = CPML.load(val, instance)
                instance.cpml.append(cpml)
                
        instance.load_data(g)

        if set_dt:
            instance.set_dt(self.dt, init=False)
        else:
            instance.dt = g.attrs['dt']

        instance.te = g.attrs['te']
        instance.th = g.attrs['th']

        return instance


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



    def save(self, g):
        super(CylindricalLangevin, self).save(g)
        g.attrs['nspecies'] = self.nspecies

        g.create_dataset('nu', data=self.nu, compression='gzip')
        g.create_dataset('wp', data=self.wp, compression='gzip')

        g.create_dataset('j', data=self.j, compression='gzip')


    def load_data(self, g):
        super(CylindricalLangevin, self).load_data(g)

        self.nu[:] = array(g['nu'])
        self.wp[:] = array(g['wp'])
        self.j[:] = array(g['j'])

        self.interpolate_e()
        
    @staticmethod
    def load(g, set_dt=False, c=None):
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
        for key, val in g.iteritems():
            if key[:5] == 'cpml_':
                cpml = CylindricalCPML.load(val, instance)
                instance.cpml.append(cpml)
                
        instance.load_data(g)

        if set_dt:
            instance.set_dt(self.dt, init=False)
        else:
            instance.dt = g.attrs['dt']

        instance.te = g.attrs['te']
        instance.th = g.attrs['th']

        return instance

def main():
    import pylab
    from matplotlib.colors import LogNorm
        
    L = 6.
    f = 278. * co.mega
    lmbd = co.c / f
    N = 100
    dt = 0.01 * (1 / f)
    n_cpml = 10
    nsave = 10
    
    box = Box(-L, L, -L, L, -L, L)
    dim = Dimensions(N, N, N)
    
    def source(t):
        return sin(2 * pi * t * f)

    nspecies = 2
    z = linspace(0, 1000, N + 1)[newaxis, newaxis, :]
    
    nu = zeros((1, 1, N + 1, nspecies))
    nu[0, 0, :, 0] = z
    nu[0, 0, :, 1] = exp(-z)
    
    wp = 1.0 * ones((N + 1, N + 1, N + 1, nspecies))
    wb = zeros((nspecies, 3))

    wb[0, :] = array([4.0, 2.0, 3.0])
    wb[1, :] = array([1.0, 2.0, 3.0])
    
    sim = Langevin(box, dim, nspecies, nu, wp, wb)
    sim.add_cpml_boundaries(n_cpml)
    sim.set_dt(dt)

    for i in xrange(10000):
        t = i * dt
        sim.update_e()
        sim.ez.v[N / 2, N / 2, N / 2] = source(t)
        sim.update_h()
        
        if 0 == (i % nsave):
            print i / nsave, t
            pylab.clf()
            pylab.pcolor(sim.xf, sim.yf, abs(sim.ez.v[:, :, N / 2].T),
                         vmin=1e-8, vmax=1,
                         norm=LogNorm())
            pylab.colorbar()

            pylab.xlim([sim.xf[0], sim.xf[-1]])
            pylab.ylim([sim.yf[0], sim.yf[-1]])

            pylab.xlabel("x [m]")
            pylab.ylabel("y [m]")

            pylab.savefig("3d_ez-%.4d.png" % (i / nsave))
    

if __name__ == '__main__':
    main()
