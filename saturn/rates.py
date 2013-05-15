from numpy import *
import scipy.constants as co
import pylab

def load_rates(infile='saturn_rates.dat'):
    en, ih2, ihe = loadtxt(infile, unpack=True,
                           usecols=(1, 17, 21))

    return en, ih2, ihe

def main():
    en, ih2, ihe = load_rates()

    effect = 0.93 * ih2 + 0.03 * ihe
    savetxt("ionization.dat", c_[en, effect])


if __name__ == '__main__':
    main()
