import sys
import os
from warnings import warn
from collections import namedtuple
import glob
import time

from numpy import *
import scipy.constants as co
import pylab
from matplotlib.colors import LogNorm
import h5py
from langevin import CylindricalLangevin
from fdtd import CylBox, CylDimensions, avg

from contexttimer import ContextTimer
X, Y, Z = 0, 1, 2
R, Z_, PHI = 0, 1, 2

Td = 1e-17 * co.centi**2

def main():
    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("-p", "--profile", dest="profile",
                      help="Profile to plot (what variables)?", 
                      action="store", default="logreduced")

    parser.add_option("-W", "--watch", dest="watch",
                      help="Watch the directory for .h5 files to appear", 
                      action="store_true", default=False)

    parser.add_option("-w", "--wait", dest="wait",
                      help="Wait for files to appear", 
                      action="store_true", default=False)

    parser.add_option("-z", "--zlim", dest="zlim",
                      help="Range of z to plot", 
                      type=str, default=None)

    parser.add_option("-r", "--rlim", dest="rlim",
                      help="Range of r to plot", 
                      type=str, default=None)

    (opts, args) = parser.parse_args()
    reduce_res = True

    if opts.zlim is not None:
        zlim = [float(s) * co.kilo for s in opts.zlim.split(':')]
        reduce_res = False
    else:
        zlim = None

    if opts.rlim is not None:
        rlim = [float(s) * co.kilo for s in opts.rlim.split(':')]
        reduce_res = False
    else:
        rlim = None


    process_files(args, opts.profile, wait=opts.wait,
                  zlim=zlim, rlim=rlim, reduce_res=reduce_res)
    if opts.watch:
        w = Watcher()
        w.run()


class Watcher(object):
    def __init__(self):
        self.done = set()
        self.todo = []
        
    def dispatch(self):
        process_files(self.todo)
        self.done.update(self.todo)
        self.todo = []
        
    def update(self):
        self.todo.extend(item for item in glob.iglob("*.h5")
                         if not item in self.done)

    def run(self):
        while True:
            self.update()
            self.dispatch()
            time.sleep(20)


def wait_for(fname):
    printed = False
    while not os.path.exists(fname):
        if not printed:
            print "Waiting for `%s'..." % fname
            printed = True

        time.sleep(10)

    # We allow an aditional 3 secs for finishing the writing
    if printed:
        time.sleep(4)

def process_files(fnames, profiles, wait=False, **kwargs):
    for fname in fnames:
        if wait:
            wait_for(fname)

        with ContextTimer("Loading %s" % fname):
            fp = h5py.File(fname)
            siml = CylindricalLangevin.load(fp)
            fp.close()
            
        for profile in profiles.split(','):
            ofile = fname.replace('.h5', '-%s.png' % profile)
            with ContextTimer("Writing %s" % ofile):
                panel(siml, profile, ofile, **kwargs)

        
def panel(sim, profile, ofile, mun=2e24, **kwargs):
    pylab.figure(figsize=(20, 8))
    pylab.clf()

    pylab.subplots_adjust(left=0.065, right=0.98, wspace=0.07, top=0.95)
    
    jr = sum(avg(sim.j_(R), axis=R), axis=-1)
    jz = sum(avg(sim.j_(Z_), axis=Z_), axis=-1)
    jphi = sum(avg(sim.j_(PHI), axis=PHI), axis=-1)
    eabs = sqrt(sum(sim.e**2, axis=-1))
    En = (eabs * co.elementary_charge / 
          (mun * sim.nu[:, :, 0] * co.electron_mass))
    En[:] = where(isfinite(En), En, 0.0)

    logargs = dict(vmin=1e-2, vmax=1e4, norm=LogNorm())
    freelogargs = dict(norm=LogNorm())


    VProfile = namedtuple("VProfile", "var label args")
    profiles = {
        "fields": [VProfile(sim.hphi.v.T, "$H_\phi$ [SI]", {}),
                   VProfile(sim.ez.v.T, "$E_z$ [V/m]", {}),
                   VProfile(sim.hr.v.T, "$H_r$ [SI]", {}),
                   VProfile(sim.er.v.T, "$E_r$ [V/m]", {})],

        "logfields": [VProfile(abs(sim.hphi.v.T), "$H_\phi$ [SI]", logargs),
                      VProfile(abs(sim.ez.v.T), "$E_z$ [V/m]", logargs),
                      VProfile(abs(sim.hr.v.T), "$H_r$ [SI]", logargs),
                      VProfile(abs(sim.er.v.T), "$E_r$ [V/m]", logargs)],

        "freelogfields": [VProfile(abs(sim.hphi.v.T), "$H_\phi$ [SI]", 
                                   freelogargs),
                          VProfile(abs(sim.ez.v.T), "$E_z$ [V/m]", 
                                   freelogargs),
                          VProfile(abs(sim.hr.v.T), "$H_r$ [SI]", 
                                   freelogargs),
                          VProfile(abs(sim.er.v.T), "$E_r$ [V/m]", 
                                   freelogargs)],

        "reduced": [VProfile(abs(jz.T), "$J_z$ [A/m$^\\mathdefault{2}$]", 
                             {}),
                    VProfile(abs(jz.T), "$J_z$ [A/m$^\\mathdefault{2}$]", 
                             {}),
                    VProfile(En.T / Td,  "$E/N$ [Td]", {}),
                    VProfile(abs(sim.er.v.T), "$E_z$ [V/m]", {})],

        "logreduced": [VProfile(abs(jz.T), "$J_z$ [A/m$^\\mathdefault{2}$]", 
                                logargs),
                       VProfile(abs(sim.er.v.T), "$E_r$ [V/m]", logargs),
                       VProfile(En.T / Td,  "$E/N$ [Td]", logargs),
                       VProfile(abs(sim.ez.v.T), "$E_z$ [V/m]", logargs)]}


    print ""
    print "\teabs : ", amin(eabs), amax(eabs)
    
    pylab.title("t = %.4d $\mu$s" % (sim.te / co.micro))

    for i, vp in enumerate(profiles[profile]):
        pylab.subplot(2, 2, i + 1)
        args = kwargs.copy()
        args.update(vp.args)

        plot(sim, vp.var, label=vp.label, **args)

    try:
        pylab.savefig(ofile)
    except (ValueError, AttributeError) as e:
        warn("Error saving the figure")


def plot(sim, var, label=None, rlim=None, zlim=None, reduce_res=True,
         **kwargs):
    rf = sim.rf
    zf = sim.zf

    if rlim is not None:
        flt = logical_and(rf >= rlim[0], rf <= rlim[1])
        rf = rf[flt]
        var = var[:, flt]

    if zlim is not None:
        flt = logical_and(zf >= zlim[0], zf <= zlim[1])
        zf = zf[flt]
        var = var[flt, :]

    if reduce_res:
        while len(rf) > 600 or len(zf) > 600:
            rf = rf[::2]
            zf = zf[::2]
            var = var[::2, ::2]
            warn("Reducing resolution to %dx%d" % (len(rf), len(zf)))
            
        
    pylab.pcolor(rf / co.kilo,
                 zf / co.kilo,
                 var, **kwargs)
    try:
        c = pylab.colorbar()
        c.set_label(label)
    except (ValueError, AttributeError) as e:
        warn("Invalid values while plotting ez")
        
    if rlim is None:
        rlim = [rf[0], rf[-1]]

    if zlim is None:
        zlim = [zf[0], zf[-1]]

    pylab.xlim([v / co.kilo for v in rlim])
    pylab.ylim([v / co.kilo for v in zlim])

    pylab.ylabel("z [km]")
    pylab.xlabel("r [km]")


if __name__ == '__main__':
    main()
