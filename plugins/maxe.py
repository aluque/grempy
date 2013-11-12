import os.path

from numpy import *
import scipy.constants as co
from scipy.interpolate import splrep, splev

from plugin import Plugin

class MaxE(Plugin):        
    def initialize(self, sim):
        # The deposited energy is evaluated at the center of the cells,
        # i.e. at the same locations as j
        sim.maxe = zeros((sim.dim.nr + 1, sim.dim.nz + 1))


    def update_h(self, sim):
        # Having independent plug-ins has some disadvantages.  One of
        # them is that we cannot re-use calculations in other plug-ins.
        # |E| is also calculated in other plug-ins.  This is a clear
        # optimization oportunity.

        eabs = sqrt(sum(sim.e**2, axis=2))
        eabs = where(isfinite(eabs), eabs, 0)

        sim.maxe[:, :] = where(eabs > sim.maxe, eabs, sim.maxe)


    def save(self, sim, g):
        g.create_dataset('maxe', data=sim.maxe, compression='gzip')


    def load_data(self, sim, g):
        sim.maxe[:, :] = array(g['maxe'])

        
