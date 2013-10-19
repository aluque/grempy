""" Simulation of a lightning EMP using FDTD. """

import sys, os
from os.path import splitext
import argparse

from numpy import *
import scipy.constants as co
import h5py

from param import ParamContainer, param, positive, contained_in

from langevin import CylindricalLangevin
from fdtd import CylBox, CylDimensions, avg
from contexttimer import ContextTimer
R, Z_, PHI = 0, 1, 2


class Parameters(ParamContainer):
    @param(default=os.path.expanduser("~/projects/em/"))
    def input_dir(s):
        """ The directory that contains extra input files. """
        return s

    @param(default='')
    def electron_density_file(s):
        """ A file to read the electron density from in h[km] n[cm^3]. """
        return s

    @param(default='')
    def gas_density_file(s):
        """ A file to read the gas density from in h[km] n[cm^3]. """
        return s

    @param(default='')
    def ionization_rate_file(s):
        """ A file to read the ionization rate from in E/n[Td] k[m^3 s^-1]. """
        return s

    @param(positive, default=0.1 * co.micro)
    def dt(s):
        """ The time step in seconds. """
        return float(s)

    @param(positive, default=60 * co.micro)
    def output_dt(s):
        """ The time between savings in seconds. """
        return float(s)

    @param(positive, default=60 * co.micro)
    def end_t(s):
        """ The final simulation time."""
        return float(s)
    
    @param(positive, default=1e3)
    def Q(s):
        """ The Charge transferred in C."""
        return float(s)
    
    @param(positive, default=50 * co.micro)
    def tau_r(s):
        """ Rise time of the stroke in seconds"""
        return float(s)

    @param(positive, default=500 * co.micro)
    def tau_f(s):
        """ Fall (decay) time of the stroke in seconds"""
        return float(s)
    
    @param(positive, default=3000)
    def r(s):
        """ Radius of the simulation domain in km. """
        return float(s)

    @param(default=-1000)
    def z0(s):
        """ Lower boundary of the simulation in km. """
        return float(s)

    @param(default=1100)
    def z1(s):
        """ Upperr boundary of the simulation in km. """
        return float(s)
    
    @param(positive, default=300)
    def r_cells(s):
        """ Number of cells in the r direction."""
        return int(s)

    @param(positive, default=110)
    def z_cells(s):
        """ Number of cells in the z direction."""
        return int(s)
    
    @param(positive, default=1.2e24)
    def mu_N(s):
        """ Mobility times N in SI units. """
        return float(s)

    @param(positive, default=10)
    def r_source(s):
        """ With of the source in km. """
        return float(s)

    @param(default=-160)
    def z0_source(s):
        """ Lower edge of the source in km. """
        return float(s)

    @param(default=-160)
    def z1_source(s):
        """ Upper edge of the source in km. """
        return float(s)

    @param(positive, default=10)
    def ncpml(s):
        """ Number of cells in the convoluted perfectly matching layers."""
        return int(s)

    @param(default=20)
    def dens_update_lower(s):
        """ Densities will not be updated below this threshold in km.  This is to avoid the effects of high fields close to the source. """
        return int(s)

    @param(default=0)
    def lower_boundary(s):
        """ Lower boundary condition.  Use 0 for a cpml, != 0 for an electrode . """
        return int(s)

    @param(default=[])
    def track_r(s):
        """ List of r-values to track. """
        return array([float(x) for x in s])

    @param(default=[])
    def track_z(s):
        """ List of z-values to track. """
        return array([float(x) for x in s])

    @param(default=[])
    def plugins(s):
        """ List of plugins for the simulation. """
        return list(s)


def j_source(t, j0, tau_r, tau_f):
    if t < tau_r:
        return j0 * t / tau_r
    else:
        return j0 * exp(-(t - tau_r)**2 / tau_f**2)


def peak(t, A, tau, m):
    return m * A / (tau * (m - 1)) * (exp(-t / tau) - exp(-m * t / tau))


def import_object(s):
    """ Tanslates module.object into __import__("module").getattr("object")
    """
    splitted = s.split('.')
    mod, obj = '.'.join(splitted[:-1]), splitted[-1]
    plugins = __import__('plugins.%s' % mod)

    return getattr(getattr(plugins, mod), obj)
    

def main():
    # == READ PARAMETERS ==
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parameter input file")
    parser.add_argument("-o", "--output", help="Output file", default=None)
    args = parser.parse_args()
    ofile = args.output or (splitext(args.input)[0] + '.h5')

    params = Parameters()
    params.file_load(args.input)

    plugins = [import_object(s)(params) for s in params.plugins]

    
    # == INIT THE SIMULATION INSTANCE ==
    box = CylBox(0, params.r * co.kilo, 
                 params.z0 * co.kilo, 
                 params.z1 * co.kilo)

    dim = CylDimensions(params.r_cells, params.z_cells)

    z = linspace(params.z0 * co.kilo, params.z1 * co.kilo, dim.nz + 1)

    sim = CylindricalLangevin(box, dim)

    # == LOAD FILES ==
    sim.load_ngas(os.path.join(params.input_dir, params.gas_density_file))
    sim.load_ne(os.path.join(params.input_dir, params.electron_density_file))
    sim.load_ionization(os.path.join(params.input_dir, 
                                     params.ionization_rate_file))
    sim.set_mun(params.mu_N)


    # == SET BOUNDARY CONDITIONS ==
    flt = ones((2, 2))
    if params.lower_boundary != 0:
        # Set the lower boundary as a perfect conductor:
        flt[Z_, 0] = False
        
        # We also have to set a Neumann bc in hphi for a conductor
        sim.hphi.neumann_axes.append((Z_, 0, 1))

    
    flt[R, 0] = False
        
    sim.add_cpml_boundaries(params.ncpml, filter=flt)
    

    # == SET SOURCE ==
    # We set the sources by using a filter in x and y
    rsource = params.r_source * co.kilo
    z0source = params.z0_source * co.kilo
    z1source = params.z1_source * co.kilo

    r2 = sim.rf**2
    source_s = pi * sim.rf[r2 <= rsource**2][-1]**2
    source_zflt = logical_and(sim.zf[newaxis, :] <= z1source, 
                              sim.zf[newaxis, :] >= z0source)
    
    si0, si1 = [nonzero(source_zflt)[1][i] for i in 0, -1]

    m = params.tau_f / params.tau_r
    

    # == PREPARE H5DF OUTPUT ==
    fp = h5py.File(ofile, 'w')
    params.h5_dump(fp)
    sim.save_global(fp)

    gsteps = fp.create_group('steps')
    gtrack = fp.create_group('track')


    # == PREPARE THE MAIN LOOP ==
    sim.set_dt(params.dt)
    sim.dens_update_lower = params.dens_update_lower * co.kilo
    insteps = 10
    nsave = int(params.output_dt / (insteps * params.dt))
    t = 0
    for p in plugins:
        p.initialize(sim)

    # == THE MAIN LOOP ==
    for i in xrange(int(params.end_t / (insteps * params.dt))):
        with ContextTimer("t = %f ms" % (t / co.milli)):
            for j in xrange(insteps):
                sim.update_e()
                for p in plugins:
                    p.update_e(sim)

                sim.update_h()
                sim.j[:, si0:si1 + 1, Z_] = \
                    (exp(-r2/rsource**2) 
                     * peak(sim.th, params.Q / source_s, 
                            params.tau_f, m))[:, newaxis]
                for p in plugins:
                    p.update_h(sim)

                t += params.dt
        
        # Save a coarser grid with higher time resolution
        if len(params.track_r) > 0 and len(params.track_z) > 0:
            step = "%.6d" % i
            g = gtrack.create_group(step)
            sim.track(g, params.track_r * co.kilo, params.track_z * co.kilo)
            fp.flush()
            
        if 0 == ((i - 1) % nsave):
            with ContextTimer("Saving step %d" % (i / nsave)):
                step = "%.4d" % (i / nsave)

                g = gsteps.create_group(step)
                sim.save(g)
                for p in plugins:
                    p.save(sim, g)

                fp.flush()
            

if __name__ == '__main__':
    main()

    
