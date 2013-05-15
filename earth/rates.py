from numpy import *
import scipy.constants as co
import pylab

def load_rates(infile='air.dat'):
    en, in2, att, io2 = loadtxt(infile, skiprows=44, unpack=True,
                                      usecols=(1, 27, 30, 44))

    return en, in2, att, io2

def main():
    en, in2, att, io2 = load_rates()

    effect = 0.8 * in2 + 0.2 * io2 - 0.2 * att
    savetxt("ionization.dat", c_[en, effect])


if __name__ == '__main__':
    main()
