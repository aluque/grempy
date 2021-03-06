

import sys
import os
from warnings import warn
import logging
import functools
from pprint import pprint
import time

from numpy import *
import scipy.constants as co
import h5py
try:
    import matplotlib
    import pylab
    from matplotlib.colors import LogNorm
    import cmaps
except ImportError:
    # There are a few things that we can do without these libraries,
    # so we allow an exception to be raised later
    warn("Unable to import matplotlib")

from langevin import CylindricalLangevin
from fdtd import avg
from em import Parameters, import_object


X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2
import functools

matplotlib.rc('font', size=22)

logger = logging.getLogger('em')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('plot.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(message)s",
                              "%a, %d %b %Y %H:%M:%S")
fh.setFormatter(formatter)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)
logging.captureWarnings(True)

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="HDF5 input file")
    parser.add_argument("var", 
                        nargs='?',
                        default=None,
                        help="Variable to plot")

    parser.add_argument("step", 
                        nargs='?',
                        help="Step to plot ('all' and 'latest' accepted)",
                        default='latest')


    parser.add_argument("-o", "--output",
                        help="Output file (may contain {rid} {step} and {var})", 
                        action='store', default='{rid}_{step}_{var}.png')

    parser.add_argument("-d", "--outdir",
                        help="Output directory (may contain {rid} and {var})", 
                        action='store', default='{rid}')

    parser.add_argument("--list", "-l",
                        help="List all the simulation steps", 
                        action='store_true', default=False)

    parser.add_argument("--parameters", "-p",
                        help="Show the simulation parameters", 
                        action='store_true', default=False)

    parser.add_argument("--log", help="Logarithmic time scale", 
                        action='store_true', default=False)

    parser.add_argument("--show", help="Interactively show the figure", 
                        action='store_true', default=False)

    parser.add_argument("--zlim", help="Limits of the z scale (z0:z1)", 
                        action='store', default=None)

    parser.add_argument("--rlim", help="Limits of the r scale (r0:r1)", 
                        action='store', default=None)

    parser.add_argument("--clim", help="Limits of the color scale (c0:c1)", 
                        action='store', default=None)

    parser.add_argument("--cmap", help="Use a dynamic colormap", 
                        action='store', default='invhot')

    parser.add_argument("--figsize", help="width height of the figure", 
                        type=float, action='store', nargs=2, default=[12, 6])

    parser.add_argument("--vars", 
                        help="Print a list available variables to plot", 
                        action='store_true', default=False)
    
    return parser

def main():
    logger.info(" ".join(sys.argv))

    parser = get_parser()
    args = parser.parse_args()

    if args.vars:
        list_vars()
        sys.exit()

    fp = h5py.File(args.input)
    all_steps = list(fp['steps'].keys())

    if args.step == 'all':
        steps = all_steps
    elif args.step == 'latest':
        steps = [all_steps[-1]]
    else:
        steps = [args.step]

    rid = os.path.splitext(args.input)[0]

    params = Parameters()
    params.h5_load(fp)

    plugins = [import_object(s)(params, load_files=False) 
               for s in params.plugins]

    if args.parameters:
        dump_params(fp)

    if args.list:
        list_steps(fp)


    if args.show and len(steps) > 1:
        logger.error(
            "For your own good, --show is incompatible with more than 1 plot.")
        sys.exit(-1)

    if args.var is None:
        sys.exit(0)

    if len(steps) > 1:
        logger.info("Plotting %d steps." % len(steps))

    outdir = args.outdir.format(rid=rid, var=args.var)
    try:
        os.mkdir(outdir)
    except OSError:
        pass


    for step in steps:
        sim = CylindricalLangevin.load(fp, step)
        for p in plugins:
            p.initialize(sim)
            p.load_data(sim, fp['steps/%s' % step])

        pylab.figure(figsize=args.figsize)
        pylab.clf()
        f = VAR_FUNCS[args.var]
        v = f(sim, params)
        plot(sim, v, args, label=f.__doc__)

        if not args.show:
            ofile = args.output.format(step=step, var=args.var, rid=rid)
            ofile = os.path.join(outdir, ofile)
            pylab.savefig(ofile)
            pylab.close()
            logger.info("File '%s' saved" % ofile)


    if args.show:
        pylab.show()


def dump_params(fp):
    pprint(dict((str(k), v) for k, v in fp.attrs.items()))

def list_vars():
    for vname, func in VAR_FUNCS.items():
        print("{}: {}".format(vname, func.__doc__))


def list_steps(fp):
    all_steps = list(fp['steps'].keys())
    for step in all_steps:
        g = fp['steps/%s' % step]
        print("%s    te = %6.5g ms  [%s]" % (step, g.attrs['te'] / co.milli, 
                                             time.ctime(g.attrs['timestamp'])))

VAR_FUNCS = {}
def plotting_var(func):
    VAR_FUNCS[func.__name__] = func
    return func

@plotting_var
def jr(sim, params):
    "$j_r [$\\mathdefault{A/m^2}$]"
    return sum(avg(sim.j_(R), axis=R), axis=-1)

@plotting_var
def jz(sim, params):
    "$j_z [$\\mathdefault{A/m^2}$]"
    return avg(sim.j_(Z_), axis=Z_)

@plotting_var
def jphi(sim, params):
    return sum(avg(sim.j_(PHI), axis=PHI), axis=-1)

@plotting_var
def er(sim, params):
    "$|E_r|$ [V/m]"
    return sim.er.v

@plotting_var
def ez(sim, params):
    "$|E_z|$ [V/m]"
    return sim.ez.v

@plotting_var
def eabs(sim, params):
    "Electric field $|E|$ [V/m]"
    return sqrt(sum(sim.e**2, axis=-1))

@plotting_var
def max_eabs(sim, params):
    "Max. Electric field $|E|$ [V/m]"
    return sim.maxe

Td = 1e-17 * co.centi**2
@plotting_var
def en(sim, params):
    "Reduced Field $E/N$ [Td]"
    mun = params.mu_N
    En = eabs(sim, params) / sim.ngas
    En[:] = where(isfinite(En), En, 0.0)

    return En / Td


@plotting_var
def max_en(sim, params):
    "Max. Reduced Field $E/N$ [Td]"
    mun = params.mu_N
    En = max_eabs(sim, params) / sim.ngas
    En[:] = where(isfinite(En), En, 0.0)

    return En / Td

@plotting_var
def ne(sim, params):
    "Electron density [$\\mathdefault{m^{-3}}$]"
    return sim.ne

@plotting_var
def ne_cm(sim, params):
    "Electron density [$\\mathdefault{cm^{-3}}$]"
    return sim.ne / co.centi**-3


@plotting_var
def fulcher(sim, params):
    "Integrated Fulcher photon emissions [$\\mathdefault{m^{-3}}$]"
    return sim.nphotons[:, :, 0]


@plotting_var
def cont(sim, params):
    "Integrated continuum photon emissions [$\\mathdefault{m^{-3}}$]"
    return sim.nphotons[:, :, 1]

@plotting_var
def photons(sim, params):
    "Integrated photon emissions [$\\mathdefault{m^{-3}}$]"
    # Let's calculate here the total of photons and inform the user.
    photons = sum(sim.nphotons[:, :, :], axis=2)
    total = sim.dr * sim.dz * sum(2 * pi * photons * sim.rf[:, newaxis])
    print("Total number of photons: %g" % total)
    return photons

@plotting_var
def photons_cm(sim, params):
    "Integrated photon emissions [$\\mathdefault{cm^{-3}}$]"
    return photons(sim, params) / co.centi**-3


@plotting_var
def energy(sim, params):
    "Deposited energy density [$\\mathdefault{J m^{-3}}$]"
    return sim.energy[:, :]

@plotting_var
def energy_cm(sim, params):
    "Deposited energy density [$\\mathdefault{J cm^{-3}}$]"
    return sim.energy[:, :] / co.centi**-3


def plot(sim, var, args, label=None, reduce_res=True):
    rf = sim.rf
    zf = sim.zf
    
    if args.rlim is not None:
        rlim = [float(x) * co.kilo for x in args.rlim.split(':')]
        flt = logical_and(rf >= rlim[0], rf <= rlim[1])
        rf = rf[flt]
        var = var[flt, :]

    if args.zlim is not None:
        zlim = [float(x) * co.kilo for x in args.zlim.split(':')]
        flt = logical_and(zf >= zlim[0], zf <= zlim[1])
        zf = zf[flt]
        var = var[:, flt]

    if reduce_res:
        while len(rf) > 600 or len(zf) > 600:
            rf = rf[::2]
            zf = zf[::2]
            var = var[::2, ::2]
            warn("Reducing resolution to %dx%d" % (len(rf), len(zf)))
            
    plot_args = {}
    if args.log:
        plot_args['norm'] = LogNorm()
        
    if args.clim:
        clim = [float(x) for x in args.clim.split(':')]
        plot_args['vmin'] = clim[0]
        plot_args['vmax'] = clim[1]

    if args.cmap:
        cmap = cmaps.get_colormap(args.cmap, dynamic=False)
        vmin = plot_args.get('vmin', nanmin(var.flat))
        vmax = plot_args.get('vmax', nanmax(var.flat))

        #cmap.center = -vmin / (vmax - vmin)
        plot_args['cmap'] = cmap

    dr = rf[1] - rf[0]
    dz = zf[1] - zf[0]
    total = (2 * pi * dr * dz * sum(var * rf[:, newaxis]))
    print("Total = %g" % (total))

    pylab.pcolor(rf / co.kilo,
                 zf / co.kilo,
                 var.T, **plot_args)
    try:
        c = pylab.colorbar()
        c.set_label(label)
    except (ValueError, AttributeError) as e:
        warn("Invalid values while plotting ez")
        
    if args.rlim is None:
        rlim = [rf[0], rf[-1]]

    if args.zlim is None:
        zlim = [zf[0], zf[-1]]
    
    pylab.text(0.975, 0.025, "t = %.2f ms" % (sim.te / co.milli),
               color="#883333",
               ha='right', va='bottom', size='x-large',
               transform=pylab.gca().transAxes)

    pylab.axis('scaled')
    pylab.xlim([v / co.kilo for v in rlim])
    pylab.ylim([v / co.kilo for v in zlim])

    pylab.ylabel("z [km]")
    pylab.xlabel("r [km]")
    #pylab.tight_layout()

if __name__ == '__main__':
    main()
