""" Simulation of a lightning EMP using FDTD. """

import sys, os
from os.path import splitext
import argparse

from numpy import *
import scipy.constants as co
from scipy.interpolate import interp1d, splrep, splev
import h5py

from param import ParamContainer, param, positive, contained_in

from langevin import CylindricalLangevin
from h2radiative import Radiative2d
from fdtd import CylBox, CylDimensions, avg
from contexttimer import ContextTimer
R, Z_, PHI = 0, 1, 2

CLASSES = {'Radiative2d': Radiative2d,
           'CylindricalLangevin': CylindricalLangevin}

class Parameters(ParamContainer):
    @param(default='CylindricalLangevin')
    def simulation_class(s):
        """ Class of the simulation """
        return s

    @param(default=os.path.expanduser("~/projects/em/"))
    def input_dir(s):
        """ The directory that contains extra input files. """
        return s

    @param()
    def electron_density_file(s):
        """ A file to read the electron density from in h[km] n[cm^3]. """
        return s

    @param()
    def gas_density_file(s):
        """ A file to read the gas density from in h[km] n[cm^3]. """
        return s

    @param()
    def rates_file(s):
        """ File with reaction rates for the radiative model. """
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

    @param(default=0)
    def lower_boundary(s):
        """ Lower boundary condition.  Use 0 for a cpml, != 0 for an electrode . """
        return int(s)

def ne(z, fname):
    """ Loads the electron density profile and interpolates it into z. """
    iri = loadtxt(fname)
    h, n = iri[:, 1] * co.kilo, iri[:, 0] * co.centi**-3

    #ipol = interp1d(h, log(n), bounds_error=False, fill_value=-inf)
    #n2 = exp(ipol(z))

    tck = splrep(h, log(n), k=1)
    n2 = exp(splev(z, tck))

    return n2


def nt(z, fname):
    """ Loads the density of neutrals and interpolates into z. """
    atm = loadtxt(fname)
    h = atm[:, 0] * co.kilo
    n = atm[:, 1] * co.centi**-3

    ipol = interp1d(h, log(n), bounds_error=False, fill_value=-inf)
    ni = exp(ipol(z))

    return ni


def j_source(t, j0, tau_r, tau_f):
    if t < tau_r:
        return j0 * t / tau_r
    else:
        return j0 * exp(-(t - tau_r)**2 / tau_f**2)


def peak(t, A, tau, m):
    return m * A / (tau * (m - 1)) * (exp(-t / tau) - exp(-m * t / tau))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Parameter input file")
    parser.add_argument("-o", "--output", help="Output file", default=None)
    args = parser.parse_args()
    ofile = args.output or (splitext(args.input)[0] + '.h5')

    params = Parameters()
    params.file_load(args.input)


    # Sets the simulation domain
    box = CylBox(0, params.r * co.kilo, 
                 params.z0 * co.kilo, 
                 params.z1 * co.kilo)

    dim = CylDimensions(params.r_cells, params.z_cells)

    z = linspace(params.z0 * co.kilo, params.z1 * co.kilo, dim.nz + 1)

    # Loads extra files containing the densities of neutrals and electrons
    # These values are interpolated at z
    nt_dens = nt(z, 
                 os.path.join(params.input_dir, params.gas_density_file))

    ne_dens = ne(z, 
                 os.path.join(params.input_dir, params.electron_density_file))


    mu = params.mu_N / nt_dens

    nu = co.elementary_charge / (mu * co.electron_mass)
    wp = sqrt(co.elementary_charge**2 
              * ne_dens / co.electron_mass / co.epsilon_0)

    nu[-params.ncpml:] = 0
    wp[-params.ncpml:] = 0
    
    # This is the relaxation time.  It puts a constraint on dt
    tau = nu / wp**2
    print "Shortest relaxation time is tau = %g s" % nanmin(tau)
    
    # Now we upgrade the parameters to the appropiate shapes.
    nu = nu[newaxis, :, newaxis]
    wp = wp[newaxis, :, newaxis]

    nspecies = 1
    frates = os.path.join(params.input_dir, params.rates_file)

    sim = CLASSES[params.simulation_class](box, dim, nspecies, nu, wp)
    if params.simulation_class == 'Radiative2d':
        sim.load_rates(frates)


    flt = ones((2, 2))
    # Set the lower boundary as a perfect conductor:
    if params.lower_boundary != 0:
        flt[Z_, 0] = False
    
    flt[R, 0] = False
    

    sim.set_densities(nt_dens[newaxis, :], ne_dens[newaxis, :])
    
    sim.add_cpml_boundaries(params.ncpml, filter=flt)
    sim.set_dt(params.dt)
    
    # We set the sources by using a filter in x and y
    rsource = params.r_source * co.kilo
    z0source = params.z0_source * co.kilo
    z1source = params.z1_source * co.kilo

    r2 = sim.rf**2
    source_s = pi * sim.rf[r2 <= rsource**2][-1]**2
    source_flt = logical_and(
        logical_and(r2[:, newaxis] <= rsource**2,
                    sim.zf[newaxis, :] <= z1source),
        sim.zf[newaxis, :] >= z0source)
    source_zflt = logical_and(sim.zf[newaxis, :] <= z1source, 
                              sim.zf[newaxis, :] >= z0source)
    
    si0, si1 = [nonzero(source_zflt)[1][i] for i in 0, -1]

    source_i = nonzero(source_zflt)
    # We add an index to set the species and one to set the coordinate (Z)
    source_i = (source_i
                + (zeros(source_i[0].shape, dtype='i'),)
                + (Z_ + zeros(source_i[0].shape, dtype='i'),))
    
    
    insteps = 100
    nsave = int(params.output_dt / (insteps * params.dt))
    t = 0
    m = params.tau_f / params.tau_r

    fp = h5py.File(ofile, 'w')
    params.h5_dump(fp)
    params.pretty_dump('pretty.yaml')
    sim.save_global(fp)

    gsteps = fp.create_group('steps')
    for i in xrange(int(params.end_t / (insteps * params.dt))):
        with ContextTimer("t = %f ms" % (t / co.milli)):
            for j in xrange(insteps):
                sim.update_e()
                sim.update_h()
                sim.j[:, si0:si1 + 1, 0, Z_] = \
                    (exp(-r2/rsource**2) 
                     * peak(sim.th, params.Q / source_s, 
                            params.tau_r, m))[:, newaxis]


                t += params.dt
                
        if 0 == ((i - 1) % nsave):
            with ContextTimer("Saving step %d" % (i / nsave)):
                step = "%.4d" % (i / nsave)

                g = gsteps.create_group(step)
                sim.save(g)
                fp.flush()
            

if __name__ == '__main__':
    main()

    
