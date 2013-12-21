import os.path

from numpy import *
import scipy.constants as co
from scipy.interpolate import splrep, splev

from .plugin import Plugin

class Energy(Plugin):        
    def initialize(self, sim):
        # The deposited energy is evaluated at the center of the cells,
        # i.e. at the same locations as j
        sim.energy = zeros((sim.dim.nr + 1, sim.dim.nz + 1))


    def update_h(self, sim):
        sim.energy[:, :] += sim.dt * sum(sim.e[:, :, :] * sim.j[:, :, :], 
                                         axis=2)


    def save(self, sim, g):
        g.create_dataset('energy', data=sim.energy, compression='gzip')


    def load_data(self, sim, g):
        sim.energy[:, :] = array(g['energy'])

        
