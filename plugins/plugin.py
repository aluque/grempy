

class Plugin(object):
    """ A class for plugins that implement things like new chemistry,
    photon counting etc.  The plguin is instantiated only once and
    then some of its methods are used as hooks at particular points of the
    simulation, receiving an instance of CylindricalLangevin as the simulation
    object.
    The methods here do nothing: they have to be superseded by sub-classes of
    plugin.
    """

    def __init__(self, sim, **kwargs):
        """ Called at the pogram initialization. """
        pass


    def initialize(self, sim):
        """ Called just before the main loop, when sim has been initialized. """
        pass


    def update_h(self, sim):
        """ Called after update_h in the main code. """
        pass


    def update_e(self, sim):
        """ Called after update_h in the main code. """
        pass


    def save(self, sim, g):
        """ Called after saving.  g is the group containing the saved step. """
        pass


    def load_data(self, sim, g):
        """ Called when loading data from group g into the simulation. """
