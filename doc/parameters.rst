Input parameters
================

``grempy`` accepts the input parameters listed here.  If a parameter is missing in 
your input file it will adopt a default value, indicated here in square brackets.

``ncpml``

    Number of cells in the convoluted perfectly matching layers. [10]


``mu_N``

    Mobility times N in SI units.  [1.2e+24]


``track_z``

    List of z-values to track.  


``plugins``

    List of plugins for the simulation.  


``track_r``

    List of r-values to track.  


``r_source``

    With of the source in km.  [10]


``gas_density_file``

    A file to read the gas density from in h[km] n[cm^3].  


``tau_f``

    Fall (decay) time of the stroke in seconds [0.0005]


``dens_update_lower``

    Densities will not be updated below this threshold in km.  This is to avoid the effects of high fields close to the source.  [20]


``output_dt``

    The time between savings in seconds.  [6e-05]


``tau_r``

    Rise time of the stroke in seconds [5e-05]


``z0_source``

    Lower edge of the source in km.  [-160]


``ionization_rate_file``

    A file to read the ionization rate from in E/n[Td] k[m^3 s^-1].  


``end_t``

    The final simulation time. [6e-05]


``Q``

    The Charge transferred in C. [1000.0]


``r_cells``

    Number of cells in the r direction. [300]


``dt``

    The time step in seconds.  [1e-07]


``z0``

    Lower boundary of the simulation in km.  [-1000]


``z1``

    Upperr boundary of the simulation in km.  [1100]


``lower_boundary``

    Lower boundary condition.  Use 0 for a cpml, != 0 for an electrode .  


``electron_density_file``

    A file to read the electron density from in h[km] n[cm^3].  


``input_dir``

    The directory that contains extra input files.  [/home/luque/projects/em/]


``z_cells``

    Number of cells in the z direction. [110]


``z1_source``

    Upper edge of the source in km.  [-160]


``r``

    Radius of the simulation domain in km.  [3000]