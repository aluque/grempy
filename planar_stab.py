""" Checking the stability of the Lee approach in 1d. """
import time

from numpy import *
import scipy.constants as co
import pylab
from saturn2d import ne, nt, peak
from fdtd import avg

# Number of cell (centers)
nc = 1000

# Number of faces
nf = nc + 1

z0, z1 = 900 * co.kilo, 1000 * co.kilo

zf = linspace(z0, z1, nf)
zc = 0.5 * avg(zf)

# The electric field is defined on cell centers
E = zeros((nc,))

# The current is defined on cell faces
J = zeros((nf,))

sz0 = 950 * co.kilo
sz1 = 960 * co.kilo
szc = (sz0 + sz1) / 2
ssigma = (sz1 - sz0) / 6
isz0, isz1 = [argmin(abs(zc - z)) for z in (sz0, sz1)]

# Here we assume a constant mu * N.  Later, we can use field-dependent
# mobility.
muN = 1.2e24                # [SI]
mu = muN / nt(zf)

nu = co.elementary_charge / (mu * co.electron_mass)
wp = sqrt(co.elementary_charge**2 * ne(zf) / co.electron_mass / co.epsilon_0)
wp2 = wp**2

# This is the relaxation time.  It puts a constraint on dt
tau = nu / wp**2

dt = 1 * co.micro
A = exp(-dt * nu)
K = (1 - A) / nu

Q = 1e3 # C.  Total charge transfered
tau_r = 50 * co.micro
tau_f = 500 * co.micro
m = tau_f / tau_r

rsource = 20 * co.kilo
source_s = pi * rsource**2

def int_step():
    """ Performs a timestep from n to n + 1.  Here it means updating E. 
    """
    E[:] = E - (0.5 * dt / co.epsilon_0) * avg(J)


def semi_step():
    """ Performs a timestep from n - 1/2 to n + 1/2.
    Here it means updating J. """

    # The b.c. here is J=0 at the boundaries
    J[1:-1] = (A[1:-1] * J[1:-1] 
               + (0.5 * co.epsilon_0 * wp2[1:-1] * K[1:-1]) * avg(E))


def main():
    f = pylab.figure(1)
    ax = pylab.subplot(1, 1, 1)

    ax.plot(zf, tau, lw=1.75, c='r')
    ax.semilogy()
    ax.axhline(dt, lw=1.8, c='k')
    pylab.show()

    pylab.ion()
    pylab.figure(2)
    ax1 = pylab.subplot(2, 1, 1)
    eline, = ax1.plot(zc, E.copy())
    ax1.set_ylim([-200000, 200000])

    ax2 = pylab.subplot(2, 1, 2)
    jline, = ax2.plot(zf, J.copy())
    ax2.set_ylim([-1, 1])
    
    
    for i in range(100000):
        t = i * dt
        int_step()
        eline.set_ydata(E.copy())

        semi_step()
        jline.set_ydata(J.copy())
        J0 = (peak(t + dt / 2, Q / source_s, tau_r, m) 
              * exp(-(zf - szc)**2 / ssigma**2))
        
        J[isz0:isz1 + 1] = J0[isz0:isz1 + 1]
        J[isz0 - 1] = 0.0
        J[isz1 + 1] = 0.0

        print(t, peak(t + dt / 2, Q / source_s, tau_r, m))
        pylab.draw()
        
        time.sleep(2)

    pylab.show()

if __name__ == '__main__':
    main()
