import numpy as np
from scipy import signal

def MakeERBFilters(fs, numChannels, lowFreq):
    # implented to python3 based on MakeERBFilters.m from AuditoryToolbox
    # This function computes the filter coefficients for a bank of
    # Gammatone filters.  These filters were defined by Patterson and
    # Holdworth for simulating the cochlea.
    #
    # The result is returned as an array of filter coefficients.  Each row
    # of the filter arrays contains the coefficients for four second order
    # filters.  The transfer function for these four filters share the same
    # denominator (poles) but have different numerators (zeros).  All of these
    # coefficients are assembled into one vector that the ERBFilterBank
    # can take apart to implement the filter.
    #
    # The filter bank contains "numChannels" channels that extend from
    # half the sampling rate (fs) to "lowFreq".

    T = 1/fs
    # Change the following parameters if you wish to use a different ERB scale.
    EarQ = 9.26449  # Glasberg and Moore Parameters
    minBW = 24.7
    order = 1

    # All of the following expressions are derived in Apple TR #35, "An
    # Efficient Implementation of the Patterson-Holdsworth Cochlear
    # Filter Bank."  See pages 33-34.
    cf = -(EarQ*minBW) + np.exp(np.arange(1, numChannels+1) *
        (-np.log(fs/2 + EarQ*minBW) + np.log(lowFreq + EarQ*minBW))/
        numChannels)*(fs/2 + EarQ*minBW)

    ERB = ((cf/EarQ) ** order + minBW**order)**(1/order)
    B = 1.019 * 2 * np.pi * ERB

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2*np.cos(2*cf*np.pi*T)/np.exp(B*T)
    B2 = np.exp(-2*B*T)

    A11 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) +
            2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2
    A12 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) -
            2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2
    A13 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) +
            2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2
    A14 = -(2*T*np.cos(2*cf*np.pi*T)/np.exp(B*T) -
            2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T)/np.exp(B*T))/2

    gain = abs((-2*np.exp(4*1j*cf*np.pi*T)*T +
                2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T*
                (np.cos(2*cf*np.pi*T) - np.sqrt(3 - 2**(3/2))*
                 np.sin(2*cf*np.pi*T)))* (-2*np.exp(4*1j*cf*np.pi*T)*T +
                 2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T*
                 (np.cos(2*cf*np.pi*T) + np.sqrt(3 - 2**(3/2)) *
                  np.sin(2*cf*np.pi*T)))*(-2*np.exp(4*1j*cf*np.pi*T)*T +
                  2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T*(np.cos(2*cf*np.pi*T) -
                  np.sqrt(3 + 2**(3/2))*np.sin(2*cf*np.pi*T)))*
                  (-2*np.exp(4*1j*cf*np.pi*T)*T + 2*np.exp(-(B*T) +
                  2*1j*cf*np.pi*T)*T*
                  (np.cos(2*cf*np.pi*T) + np.sqrt(3 + 2**(3/2))*
                  np.sin(2*cf*np.pi*T)))/
                  (-2 / np.exp(2*B*T) - 2*np.exp(4*1j*cf*np.pi*T) +
                   2*(1 + np.exp(4*1j*cf*np.pi*T))/np.exp(B*T))**4)

    allfilts = np.ones(len(cf))
    fcoefs = [A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain]

    return fcoefs
