import sys
from warnings import warn

from numpy import *
import scipy.constants as co
import pylab
from matplotlib.colors import LogNorm
import h5py
from langevin import Langevin
from fdtd import Box, Dimensions, avg

from contexttimer import ContextTimer
X, Y, Z = 0, 1, 2


def main():
    from optparse import OptionParser
    parser = OptionParser()

    (opts, args) = parser.parse_args()

    parser.add_option("-w", "--watch", dest="watch",
                      help="Wait for files to appear", 
                      action="store_true", default=False)


    process_files(args)
    if opts.watch:
        w = Watcher()
        w.run()
        
    
class Watcher(object):
    def __init__(self):
        self.done = set()
        
    def dispatch(self):
        process_files(self.todo)
        self.done.update(self.todo)
        self.todo = []
        
    def update(self):
        self.todo.extend(item if not item in self.done
                         for item in glob.iglob("*.h5"))

    def run(self):
        while True:
            self.update()
            self.dispatch()
            time.sleep(20)
        

def process_files(fnames):
    for fname in fnames:
        ofile = fname.replace('.h5', '.png')
        with ContextTimer("Loading %s" % fname):
            fp = h5py.File(fname)
            siml = Langevin.load(fp)

        with ContextTimer("Writing %s" % ofile):
            plot(siml, ofile)

def plot(sim, ofile):
    pylab.figure(figsize=(20, 16))
    pylab.clf()
    
    jx = sum(avg(sim.j_(X), axis=X), axis=-1)
    jy = sum(avg(sim.j_(Y), axis=Y), axis=-1)
    jz = sum(avg(sim.j_(Z), axis=Z), axis=-1)
    
    pylab.subplot(2, 1, 1)
    pylab.title("t = %.4d $\mu$s" % (sim.te / co.micro))
    pylab.pcolor(sim.xf[sim.dim.nx / 2:] / co.kilo,
                 sim.zf / co.kilo,
                 abs(jz[sim.dim.nx / 2:, sim.dim.ny / 2, :].T),
                 vmin=1e-16, vmax=1e-3,
                 norm=LogNorm())
    try:
        c = pylab.colorbar()
        c.set_label("$J_z$ [A/m$^\mathdefault{2}$]")
    except (ValueError, AttributeError) as e:
        warn("Invalid values while plotting jz")
        
    pylab.xlim([sim.xf[sim.dim.nx / 2] / co.kilo, sim.xf[-1] / co.kilo])
    pylab.ylim([sim.zf[0] / co.kilo, sim.zf[-1] / co.kilo])

    pylab.ylabel("z [km]")

    pylab.subplot(2, 1, 2)
    pylab.pcolor(sim.xf[sim.dim.nx / 2:] / co.kilo,
                 sim.zf / co.kilo,
                 abs(sim.ez.v[sim.dim.nx / 2:, sim.dim.ny / 2, :].T),
                 vmin=1e-6, vmax=1e4,
                 norm=LogNorm())
    try:
        c = pylab.colorbar()
        c.set_label("$E_z$ [V/m]")
    except (ValueError, AttributeError) as e:
        warn("Invalid values while plotting ez")
        
    pylab.ylabel("z [km]")
    pylab.xlabel("x [km]")

    pylab.xlim([sim.xf[sim.dim.nx / 2] / co.kilo, sim.xf[-1] / co.kilo])
    pylab.ylim([sim.zf[0] / co.kilo, sim.zf[-1] / co.kilo])


    try:
        pylab.savefig(ofile)
    except (ValueError, AttributeError) as e:
        warn("Error saving the figure")


if __name__ == '__main__':
    main()
