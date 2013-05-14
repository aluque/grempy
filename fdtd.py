""" The Yee 1966 algorithm in cartesian 3d. """

from collections import namedtuple
from numpy import *
import scipy.constants as co
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
import time

Box = namedtuple('Box', ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])
Dimensions = namedtuple('Dimensions', ['nx', 'ny', 'nz'])

CylBox = namedtuple('CylBox', ['r0', 'r1', 'z0', 'z1'])
CylDimensions = namedtuple('CylDimensions', ['nr', 'nz'])

X, Y, Z = 0, 1, 2
R, Z_ = 0, 1

class staggered(object):
    def __init__(self, shape, inface, neumann_axes=[], **kwargs):
        self.inface = inface
        self.inshape = shape
        self.bufshape = [n + 1 if s else n + 2 for n, s in zip(shape, inface)]
        self.off = [0 if s else 1 for s in inface] 
        self.full = zeros(self.bufshape, **kwargs)
        self.neumann_axes = neumann_axes
        
        # This is the array we use to access the content
        # to set the value, use .v[:]
        self.v = self.full[[s_[r:n + 1] for n, r in zip(shape, self.off)]]


    def diff(self, axis=-1, f=None):
        """ Computes the derivative along a certain axis, using the buffer
        to set the 0 b.c. without using wrong indices. """
        self.set_neumann_bc()
        
        slices = [s_[r:n + 1] for n, r in zip(self.inshape, self.off)]
        slices[axis] = s_[:]

        if f is None:
            return diff(self.full[slices], axis=axis)
        else:
            return diff(f * self.full[slices], axis=axis)

            
    def avg(self, axis=-1, f=None):
        """ Same as before but using avg instead of diff. """
        # Risk of confusion: although we call this funcion avg, from "average",
        # we do NOT include a 1/2 factor for averages.
        self.set_neumann_bc()

        slices = [s_[r:n + 1] for n, r in zip(self.inshape, self.off)]
        slices[axis] = s_[:]
        
        if f is None:
            return avg(self.full[slices], axis=axis)
        else:
            return avg(f * self.full[slices], axis=axis)
    

    def set_neumann_bc(self):
        for axis, index0, index1 in self.neumann_axes:
            if self.inface[axis]:
                continue
            slices0 = [s_[:] if i != axis else index0
                       for i in xrange(len(self.inshape))]
            slices1 = [s_[:] if i != axis else index1
                       for i in xrange(len(self.inshape))]

            self.full[slices0] = self.full[slices1]
            

    def select(self, axis, face, center):
        """ Sees if axis is a face coordinate or a center coordinate and
        returns face or center accordingly. """
        if self.inface[axis]:
            return face

        return center

    
class staggeredview(staggered):
    def __init__(self, parent, corner, shape):
        # corner here refers to cell coordinates without buffers.  We also add
        # buffers here but they actually belong to the parent grid and
        # may actually be real (non-buffer) there.
        self.inface = parent.inface
        self.inshape = shape
        self.bufshape = [n + 1 if s else n + 2
                         for n, s in zip(shape, self.inface)]

        self.off = [0 if s else 1 for s in self.inface] 
        slices = [s_[i:i + l] for i, l in zip(corner, self.bufshape)]

        self.full = parent.full[slices]
        self.v = self.full[[s_[r:n + 1] for n, r in zip(shape, self.off)]]
        self.parent = parent
        

    def set_neumann_bc(self):
        self.parent.set_neumann_bc()
    

class Cartesian3d(object):
    """ A simulation domain for the cartesian 3d Yee algorithm. """
    def __init__(self, box, dim):
        self.box = box
        self.dim = dim

        # the face and center locations
        self.xf = linspace(box.x0, box.x1, dim.nx + 1)
        self.xc = centers(self.xf)
        self.dx = self.xf[1] - self.xf[0]

        self.yf = linspace(box.y0, box.y1, dim.ny + 1)
        self.yc = centers(self.yf)
        self.dy = self.yf[1] - self.yf[0]

        self.zf = linspace(box.z0, box.z1, dim.nz + 1)
        self.zc = centers(self.zf)
        self.dz = self.zf[1] - self.zf[0]

        self.rf = [self.xf, self.yf, self.zf]
        self.rc = [self.xc, self.yc, self.zc]
        self.dr = [self.dx, self.dy, self.dz]
        
        # A list of cpml layers associated to the simulation (generally 6).
        self.cpml = []
        self.allocate_fields()
        

    def allocate_fields(self):
        """ Allocate the field arrays, with two extra cell in each dimension
        to implement the b.c. """

        self.ex = staggered(self.dim, (False, True, True))
        self.ey = staggered(self.dim, (True, False, True))
        self.ez = staggered(self.dim, (True, True, False))

        self.hx = staggered(self.dim, (True, False, False))
        self.hy = staggered(self.dim, (False, True, False))
        self.hz = staggered(self.dim, (False, False, True))


    def update_e(self):
        """ Updates E{x, y, z} from time n to n+1 assuming that the
        H{x, y, z} are up to date at time n+1/2. """
        dt = self.dt
        
        self.ex.v[:] = self.ex.v[:] + (dt / co.epsilon_0
                                       * (self.hz.diff(Y) / self.dy
                                          - self.hy.diff(Z) / self.dz))

        self.ey.v[:] = self.ey.v[:] + (dt / co.epsilon_0
                                       * (self.hx.diff(Z) / self.dz
                                          - self.hz.diff(X) / self.dx))

        self.ez.v[:] = self.ez.v[:] + (dt / co.epsilon_0
                                       * (self.hy.diff(X) / self.dx
                                          - self.hx.diff(Y) / self.dy))

        for cpml in self.cpml:
            cpml.update_e()

        self.te += dt

    def update_h(self):
        """ Updates H{x, y, z} from time n-1/2 to n+1/2 assuming that the
        E{x, y, z} are up to date at time n. """
        dt = self.dt

        self.hx.v[:] = self.hx.v[:] - (dt / co.mu_0
                                       * (self.ez.diff(Y) / self.dy
                                          - self.ey.diff(Z) / self.dz))

        self.hy.v[:] = self.hy.v[:] - (dt / co.mu_0
                                       * (self.ex.diff(Z) / self.dz
                                          - self.ez.diff(X) / self.dx))

        self.hz.v[:] = self.hz.v[:] - (dt / co.mu_0
                                       * (self.ey.diff(X) / self.dx
                                          - self.ex.diff(Y) / self.dy))

        for cpml in self.cpml:
            cpml.update_h()

        self.th += dt

    def forward(self, dt):
        self.update_e(dt)
        self.update_h(dt)
        

    def new_cpml(self, *args, **kwargs):
        cpml = CPML(self, *args, **kwargs)
        self.cpml.append(cpml)


    def add_cpml_boundaries(self, n, filter=None):
        """ Adds six cpml layers, one for each of the boundaries of the
        simulation. """
        
        for i in X, Y, Z:
            if filter is None or filter[i, 0]:
                self.new_cpml(i, n)

            if filter is None or filter[i, 1]:
                self.new_cpml(i, n, invert=True)


    def set_dt(self, dt, init=True):
        self.dt = dt

        # The times of the current electric and magnetic fields
        if init:
            self.te = 0
            self.th = dt / 2
        
        for cpml in self.cpml:
            cpml.set_dt(dt)
        
        
    def save(self, g):
        """ Saves the state of this grid into g.  g has to satisfy the group
        interface of h5py.  
        """
        g.attrs['dt'] = self.dt
        g.attrs['te'] = self.te
        g.attrs['th'] = self.th
        g.attrs['timestamp'] = time.time()

        g.create_dataset('box', data=self.box, compression='gzip')
        g.create_dataset('dim', data=self.dim, compression='gzip')

        g.create_dataset('ex', data=self.ex.full, compression='gzip')
        g.create_dataset('ey', data=self.ey.full, compression='gzip')
        g.create_dataset('ez', data=self.ez.full, compression='gzip')

        g.create_dataset('hx', data=self.hx.full, compression='gzip')
        g.create_dataset('hy', data=self.hy.full, compression='gzip')
        g.create_dataset('hz', data=self.hz.full, compression='gzip')
        
        for i, cpml in enumerate(self.cpml):
            g2 = g.create_group('cpml_%.3d' % i)
            cpml.save(g2)
            

    def load_data(self, g):
        self.ex.full[:] = array(g['ex'])
        self.ey.full[:] = array(g['ey'])
        self.ez.full[:] = array(g['ez'])

        self.hx.full[:] = array(g['hx'])
        self.hy.full[:] = array(g['hy'])
        self.hz.full[:] = array(g['hz'])

        for i, cpml in enumerate(self.cpml):
            g2 = g['cpml_%.3d' % i]
            cpml.load_data(g2)


class Cylindrical(object):
    """ A simulation domain for the 3d Yee algorithm in cylindical
    coordinates and with axial symmetry. """
    def __init__(self, box, dim):
        self.box = box
        self.dim = dim

        # the face and center locations
        self.rf = linspace(box.r0, box.r1, dim.nr + 1)
        self.rc = centers(self.rf)
        self.dr = self.rf[1] - self.rf[0]

        # In cyl. coordinates we also need an array r at the centers but
        # extended with two buffer cells.  We also promote it to larger
        # dimensions.
        self.rcb = r_[self.rc[0] - self.dr, self.rc, self.rc[-1] + self.dr]
            
        self.zf = linspace(box.z0, box.z1, dim.nz + 1)
        self.zc = centers(self.zf)
        self.dz = self.zf[1] - self.zf[0]

        self.rzf = [self.rf, self.zf]
        self.rzc = [self.rc, self.zc]
        self.drz = [self.dr, self.dz]

        # A list of cpml layers associated to the simulation (generally 6).
        self.cpml = []
        self.allocate_fields()
        

    def allocate_fields(self):
        """ Allocate the field arrays, with two extra cell in each dimension
        to implement the b.c. """

        neumann = [(R, 0, 1)]
        
        self.er = staggered(self.dim, (False, True), neumann_axes=neumann)
        self.ez = staggered(self.dim, (True, False), neumann_axes=neumann)
        self.ephi = staggered(self.dim, (True, True), neumann_axes=neumann)

        self.hr = staggered(self.dim, (True, False), neumann_axes=neumann)
        self.hz = staggered(self.dim, (False, True), neumann_axes=neumann)
        self.hphi = staggered(self.dim, (False, False), neumann_axes=neumann)

    
    
    def interpolator(self, v):
        """ Build and interpolator for v. """

        rbs = RectBivariateSpline(
            v.select(0, self.rf, self.rc),
            v.select(1, self.zf, self.zc),
            v.v)

        return rbs


    def update_e(self):
        """ Updates E{r, z, phi} from time n to n+1 assuming that the
        H{r, z, phi} are up to date at time n+1/2. """
        dt = self.dt
        
        self.er.v[:] = self.er.v[:] - (dt / co.epsilon_0
                            * self.hphi.diff(Z_) / self.dz)
        self.ez.v[1:, :] = self.ez.v[1:, :] + (dt / co.epsilon_0
                                               * self.hphi.diff(R,
                                                                f=self.rcb[:, newaxis])
                                               / self.dr
                                               / self.rf[:, newaxis])[1:, :]

        # See Inan & Marshall, (4.39)
        self.ez.v[0, :] = self.ez.v[0, :] + (4 * dt / co.epsilon_0 / self.dr
                                             * self.hphi.v[0, :])

        self.ephi.v[:] = self.ephi.v[:] + (dt / co.epsilon_0
                                           * (self.hr.diff(Z_) / self.dz
                                              - self.hz.diff(R) / self.dr))


        for cpml in self.cpml:
            cpml.update_e()

        self.te += dt

    def update_h(self):
        """ Updates H{x, y, z} from time n-1/2 to n+1/2 assuming that the
        E{x, y, z} are up to date at time n. """
        dt = self.dt

        self.hr.v[:] = self.hr.v[:] + (dt / co.mu_0
                                       * self.ephi.diff(Z_) / self.dz)

        self.hz.v[:] = self.hz.v[:] - (dt / co.mu_0
                                       * (self.ephi.diff(R,
                                                         f=self.rf[:, newaxis])
                                          / self.dr / self.rc[:, newaxis]))

        self.hphi.v[:] = self.hphi.v[:] + (dt / co.mu_0
                                           * (self.ez.diff(R) / self.dr
                                              - self.er.diff(Z_) / self.dz))

        for cpml in self.cpml:
            cpml.update_h()

        self.th += dt

    def forward(self, dt):
        self.update_e(dt)
        self.update_h(dt)


    def new_cpml(self, *args, **kwargs):
        cpml = CylindricalCPML(self, *args, **kwargs)
        self.cpml.append(cpml)


    def add_cpml_boundaries(self, n, filter=None):
        """ Adds six cpml layers, one for each of the boundaries of the
        simulation. """
        for i in R, Z_:
            if filter is None or filter[i, 0]:
                self.new_cpml(i, n)

            if filter is None or filter[i, 1]:
                self.new_cpml(i, n, invert=True)


    def set_dt(self, dt, init=True):
        self.dt = dt

        # The times of the current electric and magnetic fields
        if init:
            self.te = 0
            self.th = dt / 2
        
        for cpml in self.cpml:
            cpml.set_dt(dt)
        

    def save_global(self, g):
        """ Saves the global properties of the simulation into g. 
        In contrast to save, the values here do not change during a simulation.
        """
        g.create_dataset('box', data=self.box, compression='gzip')
        g.create_dataset('dim', data=self.dim, compression='gzip')


    def _save(self, g, save_cpml=True, f=None):
        """ Saves the state of this grid into g.  g has to satisfy the group
        interface of h5py.  """
        if f is None:
            f = lambda(x): x.full

        g.attrs['dt'] = self.dt
        g.attrs['te'] = self.te
        g.attrs['th'] = self.th
        g.attrs['timestamp'] = time.time()

        g.create_dataset('er', data=f(self.er), compression='gzip')
        g.create_dataset('ez', data=f(self.ez), compression='gzip')
        g.create_dataset('ephi', data=f(self.ephi), compression='gzip')

        g.create_dataset('hr', data=f(self.hr), compression='gzip')
        g.create_dataset('hz', data=f(self.hz), compression='gzip')
        g.create_dataset('hphi', data=f(self.hphi), compression='gzip')
        
        if not save_cpml:
            return

        for i, cpml in enumerate(self.cpml):
            g2 = g.create_group('cpml_%.3d' % i)
            cpml.save(g2)
            

    def save(self, g):
        self._save(g)


    def track(self, g, r, z):
        # This is not correct: for staggered grids, I am not
        # taking the closest point of some vars.  It's only for
        # output but should be fixed.
        i = around(r / self.dr).astype('i')
        j = around(z / self.dz).astype('i')
        
        def fmesh(d):
            return d.v[i[:, newaxis], j[newaxis, :]]
        
        self._save(g, save_cpml=False, f=fmesh)


    def resampled_save(self, g):
        sr = linspace(self.box.r0 + self.dr, self.box.r1 - self.dr, 
                      self.dim.nr / 100 + 1)
        sz = linspace(self.box.z0 + self.dz, self.box.z1 - self.dz, 
                      self.dim.nz / 100 + 1)

        ier = self.interpolator(self.er)
        iez = self.interpolator(self.ez)
        
        er = ier(sr, sz)
        ez = iez(sr, sz)
        
        g.create_dataset('r', data=sr, compression='gzip')
        g.create_dataset('z', data=sz, compression='gzip')
        g.create_dataset('er', data=er, compression='gzip')
        g.create_dataset('ez', data=ez, compression='gzip')


    def load_data(self, g):
        self.er.full[:] = array(g['er'])
        self.ez.full[:] = array(g['ez'])
        self.ephi.full[:] = array(g['ephi'])

        self.hr.full[:] = array(g['hr'])
        self.hz.full[:] = array(g['hz'])
        self.hphi.full[:] = array(g['hphi'])

        for i, cpml in enumerate(self.cpml):
            g2 = g['cpml_%.3d' % i]
            cpml.load_data(g2)


class CPML(object):
    """ Class for Convolutional Perfect Matching Layers.
    Inside this class we follow the evolution of the 12 pseudo-fields
    psi.  In a 3d cube we must include 6 CPMLs. """

    def __init__(self, sim, coord, n, invert=False, m=4, R0=1e-6):
        """ * coord is the coordinate (X, Y, Z) that we use to grade the CPML.
            * invert has to be true if the CPML is located at a maximum of
              coord
        """
        self.dim = Dimensions(*[sim.dim[i] - 2 * n if i != coord else n
                                for i in (X, Y, Z)])
        
        self.n = n
        self.invert = invert
        self.m = m
        self.R0 = R0
        
        # sim has to be a Cartesion3D object to take the fields from.
        self.sim = sim

        self.corner = [n for i in (X, Y, Z)]
        if not invert:
            self.corner[coord] = 0
        else:
            self.corner[coord] = sim.dim[coord] - n
            
        # the face and center locations.
        
        self.coord = coord

        # We will call u the grading coordinate, be it x, y, or z
        i = coord
        self.uf = sim.rf[i][self.corner[i]: (self.corner[i] + self.dim[i] + 1)]
        self.uc = sim.rc[i][self.corner[i]: self.corner[i] + self.dim[i]]
        self.du = sim.dr[i]
        self.nu = self.dim[i]
        self.Lu = self.nu * self.du
        
        eta = sqrt(co.mu_0 / co.epsilon_0)
        self.smax = -(m + 1) * log(R0) / (2 * eta * self.Lu)
        
        u0 = self.uf[-1 if not invert else 0]
        self.sf = self.smax * (abs(u0 - self.uf) / self.Lu)**m
        self.sc = self.smax * (abs(u0 - self.uc) / self.Lu)**m
        
        # We now upgrade sf and sc to 3-dimensional arrays using newaxis
        s = [newaxis if i != coord else s_[:] for i in (X, Y, Z)]
        self.sf = self.sf[s]
        self.sc = self.sc[s]

        self.dx, self.dy, self.dz = sim.dr
        
        self.allocate_fields()


    def allocate_fields(self):
        """ Allocate the field arrays, with two extra cell in each dimension
        to implement the b.c. """
        
        self.ex = staggeredview(self.sim.ex, self.corner, self.dim)
        self.ey = staggeredview(self.sim.ey, self.corner, self.dim)
        self.ez = staggeredview(self.sim.ez, self.corner, self.dim)

        self.hx = staggeredview(self.sim.hx, self.corner, self.dim)
        self.hy = staggeredview(self.sim.hy, self.corner, self.dim)
        self.hz = staggeredview(self.sim.hz, self.corner, self.dim)
        
        # These are the 12 psi fields.  Each one is evaluated at the
        # same locations as the corresponding field.  For example,
        # pexy and pexz represent Psi_{ex,x} and Psi_{ex,z} and are located
        # at the same places as E_{x}.
        self.pexy = staggered(self.dim, (False, True, True))
        self.pexz = staggered(self.dim, (False, True, True))
        
        self.peyx = staggered(self.dim, (True, False, True))
        self.peyz = staggered(self.dim, (True, False, True))
        
        self.pezx = staggered(self.dim, (True, True, False))
        self.pezy = staggered(self.dim, (True, True, False))

        self.phxy = staggered(self.dim, (True, False, False))
        self.phxz = staggered(self.dim, (True, False, False))

        self.phyx = staggered(self.dim, (False, True, False))
        self.phyz = staggered(self.dim, (False, True, False))

        self.phzx = staggered(self.dim, (False, False, True))
        self.phzy = staggered(self.dim, (False, False, True))


    def set_dt(self, dt):
        self.af = exp(-self.sf * dt / co.epsilon_0) - 1
        self.ac = exp(-self.sc * dt / co.epsilon_0) - 1
        self.bf = self.af + 1
        self.bc = self.ac + 1
        self.dt = dt

        # To make the multiplication by the fields easier we select
        # here variables for each field according to wether that field
        # is located in faces or in centers along the direction given by
        # coord. 
        self.bex = self.ex.select(self.coord, self.bf, self.bc)
        self.aex = self.ex.select(self.coord, self.af, self.ac)

        self.bey = self.ey.select(self.coord, self.bf, self.bc)
        self.aey = self.ey.select(self.coord, self.af, self.ac)

        self.bez = self.ez.select(self.coord, self.bf, self.bc)
        self.aez = self.ez.select(self.coord, self.af, self.ac)

        self.bhx = self.hx.select(self.coord, self.bf, self.bc)
        self.ahx = self.hx.select(self.coord, self.af, self.ac)

        self.bhy = self.hy.select(self.coord, self.bf, self.bc)
        self.ahy = self.hy.select(self.coord, self.af, self.ac)

        self.bhz = self.hz.select(self.coord, self.bf, self.bc)
        self.ahz = self.hz.select(self.coord, self.af, self.ac)


    def update_e(self):
        self.pexy.v[:] = (self.bex * self.pexy.v[:]
                          + self.aex * self.hz.diff(Y) / self.dy)
        self.pexz.v[:] = (self.bex * self.pexz.v[:]
                          + self.aex * self.hy.diff(Z) / self.dz)

        self.peyx.v[:] = (self.bey * self.peyx.v[:]
                          + self.aey * self.hz.diff(X) / self.dx)
        self.peyz.v[:] = (self.bey * self.peyz.v[:]
                          + self.aey * self.hx.diff(Z) / self.dz)

        self.pezx.v[:] = (self.bez * self.pezx.v[:]
                          + self.aez * self.hy.diff(X) / self.dx)
        self.pezy.v[:] = (self.bez * self.pezy.v[:]
                          + self.aez * self.hx.diff(Y) / self.dy)

        eps0 = co.epsilon_0
        self.ex.v[:] += ((self.dt / eps0) * (self.pexy.v[:] - self.pexz.v[:]))
        self.ey.v[:] += ((self.dt / eps0) * (self.peyz.v[:] - self.peyx.v[:]))
        self.ez.v[:] += ((self.dt / eps0) * (self.pezx.v[:] - self.pezy.v[:]))


    def update_h(self):
        self.phxy.v[:] = (self.bhx * self.phxy.v[:]
                          + self.ahx * self.ez.diff(Y) / self.dy)
        self.phxz.v[:] = (self.bhx * self.phxz.v[:]
                          + self.ahx * self.ey.diff(Z) / self.dz)

        self.phyx.v[:] = (self.bhy * self.phyx.v[:]
                          + self.ahy * self.ez.diff(X) / self.dx)
        self.phyz.v[:] = (self.bhy * self.phyz.v[:]
                          + self.ahy * self.ex.diff(Z) / self.dz)

        self.phzx.v[:] = (self.bhz * self.phzx.v[:]
                          + self.ahz * self.ey.diff(X) / self.dx)
        self.phzy.v[:] = (self.bhz * self.phzy.v[:]
                          + self.ahz * self.ex.diff(Y) / self.dy)


        mu0 = co.mu_0
        self.hx.v[:] -= ((self.dt / mu0) * (self.phxy.v[:] - self.phxz.v[:]))
        self.hy.v[:] -= ((self.dt / mu0) * (self.phyz.v[:] - self.phyx.v[:]))
        self.hz.v[:] -= ((self.dt / mu0) * (self.phzx.v[:] - self.phzy.v[:]))


    def save(self, g):
        g.attrs['coord'] = self.coord
        g.attrs['n'] = self.n
        g.attrs['invert'] = self.invert
        g.attrs['m'] = self.m
        g.attrs['R0'] = self.R0
        
        g.create_dataset('dim', data=self.dim, compression='gzip')
        g.create_dataset('corner', data=self.dim, compression='gzip')

        g.create_dataset('pexy', data=self.pexy.full, compression='gzip')
        g.create_dataset('pexz', data=self.pexz.full, compression='gzip')
        g.create_dataset('peyx', data=self.peyx.full, compression='gzip')
        g.create_dataset('peyz', data=self.peyz.full, compression='gzip')
        g.create_dataset('pezx', data=self.pezx.full, compression='gzip')
        g.create_dataset('pezy', data=self.pezy.full, compression='gzip')

        g.create_dataset('phxy', data=self.phxy.full, compression='gzip')
        g.create_dataset('phxz', data=self.phxz.full, compression='gzip')
        g.create_dataset('phyx', data=self.phyx.full, compression='gzip')
        g.create_dataset('phyz', data=self.phyz.full, compression='gzip')
        g.create_dataset('phzx', data=self.phzx.full, compression='gzip')
        g.create_dataset('phzy', data=self.phzy.full, compression='gzip')


    def load_data(self, g):
        self.pexy.full[:] = array(g['pexy'])
        self.pexz.full[:] = array(g['pexz'])
        self.peyx.full[:] = array(g['peyx'])
        self.peyz.full[:] = array(g['peyz'])
        self.pezx.full[:] = array(g['pezx'])
        self.pezy.full[:] = array(g['pezy'])
        self.phxy.full[:] = array(g['phxy'])
        self.phxz.full[:] = array(g['phxz'])
        self.phyx.full[:] = array(g['phyx'])
        self.phyz.full[:] = array(g['phyz'])
        self.phzx.full[:] = array(g['phzx'])
        self.phzy.full[:] = array(g['phzy'])

        
    @staticmethod
    def load(g, sim):
        coord = array(g.attrs['coord'])
        n = g.attrs['n']
        invert = g.attrs['invert']
        m = g.attrs['m']
        R0 = g.attrs['R0']
        
        instance = CPML(sim, coord, n, invert=invert, m=m, R0=R0)

        return instance
    

class CylindricalCPML(object):
    """ Class for Convolutional Perfect Matching Layers in cylindrical symmetry.
    """

    def __init__(self, sim, coord, n, invert=False, m=4, R0=1e-6):
        """ * coord is the coordinate (R, Z) that we use to grade the CPML.
            * invert has to be true if the CPML is located at a maximum of
              coord
        """
        self.dim = CylDimensions(*[sim.dim[i] - 2 * n if i != coord else n
                                for i in (R, Z_)])
        
        self.n = n
        self.invert = invert
        self.m = m
        self.R0 = R0
        
        # sim has to be a Cylindrical object to take the fields from.
        self.sim = sim

        self.corner = [1 for i in (R, Z_)]
        if not invert:
            self.corner[coord] = 0
        else:
            self.corner[coord] = sim.dim[coord] - n
                    
        self.coord = coord
        self.dr, self.dz = sim.dr, sim.dz

        # We will call u the grading coordinate, be it r or z
        i = coord
        self.uf = sim.rzf[i][self.corner[i]: (self.corner[i] + self.dim[i] + 1)]
        self.uc = sim.rzc[i][self.corner[i]: self.corner[i] + self.dim[i]]
        self.du = sim.drz[i]
        self.nu = self.dim[i]
        self.Lu = self.nu * self.du
        
        # We specially need the r coordinate
        self.rf = sim.rf[self.corner[R]: (self.corner[R] + self.dim[R] + 1)]
        self.rc = sim.rc[self.corner[R]: self.corner[R] + self.dim[R]]
        self.rcb = r_[self.rc[0] - self.dr, self.rc, self.rc[-1] + self.dr]
        
        eta = sqrt(co.mu_0 / co.epsilon_0)
        self.smax = -(m + 1) * log(R0) / (2 * eta * self.Lu)
        
        u0 = self.uf[-1 if not invert else 0]
        self.sf = self.smax * (abs(u0 - self.uf) / self.Lu)**m
        self.sc = self.smax * (abs(u0 - self.uc) / self.Lu)**m
        
        # We now upgrade sf and sc to 3-dimensional arrays using newaxis
        s = [newaxis if i != coord else s_[:] for i in (R, Z_)]
        self.sf = self.sf[s]
        self.sc = self.sc[s]

        self.allocate_fields()


    def allocate_fields(self):
        """ Allocate the field arrays, with two extra cell in each dimension
        to implement the b.c. """
        
        self.er = staggeredview(self.sim.er, self.corner, self.dim)
        self.ez = staggeredview(self.sim.ez, self.corner, self.dim)
        self.ephi = staggeredview(self.sim.ephi, self.corner, self.dim)

        self.hr = staggeredview(self.sim.hr, self.corner, self.dim)
        self.hz = staggeredview(self.sim.hz, self.corner, self.dim)
        self.hphi = staggeredview(self.sim.hphi, self.corner, self.dim)
        
        # These are the 12 psi fields.  Each one is evaluated at the
        # same locations as the corresponding field.  For example,
        # pexy and pexz represent Psi_{ex,x} and Psi_{ex,z} and are located
        # at the same places as E_{x}.
        self.perz = staggered(self.dim, (False, True))
        self.pezr = staggered(self.dim, (True, False))
        
        self.pephiz = staggered(self.dim, (True, True))
        self.pephir = staggered(self.dim, (True, True))
        
        self.phrz = staggered(self.dim, (True, False))
        self.phzr = staggered(self.dim, (False, True))

        self.phphir = staggered(self.dim, (False, False))
        self.phphiz = staggered(self.dim, (False, False))


    def set_dt(self, dt):
        self.af = exp(-self.sf * dt / co.epsilon_0) - 1
        self.ac = exp(-self.sc * dt / co.epsilon_0) - 1
        self.bf = self.af + 1
        self.bc = self.ac + 1
        self.dt = dt

        # To make the multiplication by the fields easier we select
        # here variables for each field according to wether that field
        # is located in faces or in centers along the direction given by
        # coord. 
        self.ber = self.er.select(self.coord, self.bf, self.bc)
        self.aer = self.er.select(self.coord, self.af, self.ac)

        self.bez = self.ez.select(self.coord, self.bf, self.bc)
        self.aez = self.ez.select(self.coord, self.af, self.ac)

        self.bephi = self.ephi.select(self.coord, self.bf, self.bc)
        self.aephi = self.ephi.select(self.coord, self.af, self.ac)

        self.bhr = self.hr.select(self.coord, self.bf, self.bc)
        self.ahr = self.hr.select(self.coord, self.af, self.ac)

        self.bhz = self.hz.select(self.coord, self.bf, self.bc)
        self.ahz = self.hz.select(self.coord, self.af, self.ac)

        self.bhphi = self.hphi.select(self.coord, self.bf, self.bc)
        self.ahphi = self.hphi.select(self.coord, self.af, self.ac)


    def update_e(self):
        self.perz.v[:] = (self.ber * self.perz.v[:]
                          + self.aer * self.hphi.diff(Z_) / self.dz)
        self.pezr.v[:] = (self.bez * self.pezr.v[:]
                          + self.aez * self.hphi.diff(R, f=self.rcb[:, newaxis])
                          / self.dr / self.rf[:, newaxis])

        self.pephiz.v[:] = (self.bephi * self.pephiz.v[:]
                          + self.aephi * self.hr.diff(Z_) / self.dz)
        self.pephir.v[:] = (self.bephi * self.pephir.v[:]
                          + self.aephi * self.hz.diff(R) / self.dr)

        eps0 = co.epsilon_0
        self.er.v[:] -= (self.dt / eps0) * self.perz.v[:]
        self.ez.v[:] += (self.dt / eps0) * self.pezr.v[:]
        self.ephi.v[:] += ((self.dt / eps0) * (self.pephiz.v[:]
                                               - self.pephir.v[:]))


    def update_h(self):
        self.phrz.v[:] = (self.bhr * self.phrz.v[:]
                          + self.ahr * self.ephi.diff(Z_) / self.dz)
        self.phzr.v[:] = (self.bhz * self.phzr.v[:]
                          + self.ahz * self.ephi.diff(R, f=self.rf[:, newaxis])
                          / self.dr / self.rc[:, newaxis])

        self.phphiz.v[:] = (self.bhphi * self.phphiz.v[:]
                          + self.ahphi * self.er.diff(Z_) / self.dz)
        self.phphir.v[:] = (self.bhphi * self.phphir.v[:]
                          + self.ahphi * self.ez.diff(R) / self.dr)


        mu0 = co.mu_0
        self.hr.v[:]   += (self.dt / mu0) * self.phrz.v[:]
        self.hz.v[:]   -= (self.dt / mu0) * self.phzr.v[:]
        self.hphi.v[:] += ((self.dt / mu0) * (self.phphir.v[:]
                                              - self.phphiz.v[:]))


    def save(self, g):
        g.attrs['coord'] = self.coord
        g.attrs['n'] = self.n
        g.attrs['invert'] = self.invert
        g.attrs['m'] = self.m
        g.attrs['R0'] = self.R0
        
        g.create_dataset('dim', data=self.dim, compression='gzip')
        g.create_dataset('corner', data=self.dim, compression='gzip')

        g.create_dataset('pezr', data=self.pezr.full, compression='gzip')
        g.create_dataset('perz', data=self.perz.full, compression='gzip')
        g.create_dataset('pephir', data=self.pephir.full, compression='gzip')
        g.create_dataset('pephiz', data=self.pephiz.full, compression='gzip')

        g.create_dataset('phzr', data=self.phzr.full, compression='gzip')
        g.create_dataset('phrz', data=self.phrz.full, compression='gzip')
        g.create_dataset('phphir', data=self.phphir.full, compression='gzip')
        g.create_dataset('phphiz', data=self.phphiz.full, compression='gzip')


    def load_data(self, g):
        self.pezr.full[:] = array(g['pezr'])
        self.perz.full[:] = array(g['perz'])
        self.pephiz.full[:] = array(g['pephir'])
        self.pephir.full[:] = array(g['pephiz'])
        self.phzr.full[:] = array(g['phzr'])
        self.phrz.full[:] = array(g['phrz'])
        self.phphir.full[:] = array(g['phphir'])
        self.phphiz.full[:] = array(g['phphiz'])

        
    @staticmethod
    def load(g, sim):
        coord = array(g.attrs['coord'])
        n = g.attrs['n']
        invert = g.attrs['invert']
        m = g.attrs['m']
        R0 = g.attrs['R0']
        
        instance = CylindricalCPML(sim, coord, n, invert=invert, m=m, R0=R0)

        return instance

        
def avg(a, n=1, axis=-1):
    """
    This is based on diff from numpy but instead of calculating differences
    it calculates sums (i.e. instead of x[1:] - x[:-1] would calculate
    x[1:] + x[:-1]
    """
    if n == 0:
        return a
    if n < 0:
        raise ValueError(
                "order must be non-negative but got " + repr(n))
    a = asanyarray(a)
    nd = len(a.shape)
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    if n > 1:
        return avg(a[slice1]+a[slice2], n-1, axis=axis)
    else:
        return a[slice1]+a[slice2]


def centers(x):
    """ Given an array of face coords, returns the center locations. """
    return 0.5 * avg(x)


def main():
    import pylab
    from matplotlib.colors import LogNorm

    L = 6.
    f = 278. * co.mega
    lmbd = co.c / f
    N = 100
    dt = 0.01 * (1 / f)
    n_cpml = 10
    
    box = Box(-L, L, -L, L, -L, L)
    dim = Dimensions(N, N, N)
    
    def source(t):
        return sin(2 * pi * t * f)

    sim = Cartesian3d(box, dim)
    nsave = 10
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
