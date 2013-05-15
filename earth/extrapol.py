from numpy import *
import pylab

z, ne = loadtxt("iri.dat", unpack=True)
h = 2.857
h0 = 60
ne0 = 1e-2

zext = linspace(0, z[0], 128)[:-1]
neext = ne0 * exp((zext - h0) / h)

savetxt("electrons.dat", c_[r_[zext, z], r_[neext, ne]])

pylab.plot(r_[neext, ne], r_[zext, z], lw=2.0)
pylab.show()
