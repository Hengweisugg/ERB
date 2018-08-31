import numpy as np
from scipy import signal
from MakeERBFilters import MakeERBFilters


def ERBFilterBank(x, fcoefs=MakeERBFilters(22050,64,100)):
	# implented to python3 based on ERBFilterBank.m from AuditoryToolbox
	# Process an input waveform with a gammatone filter bank. Design the
	# fcoefs parameter, which completely specifies the Gammatone filterbank,
	# with the MakeERBFilters function.

	# This function takes a single sound vector, and returns an array of filter
	# outputs, one output per column. The filter coefficients are computed for
	# you if you do not specify them.  The default Gammatone filter bank assumes
	# a 22050Hz sampling rate, and provides 64 filters down to 100Hz.

	if len(fcoefs) != 10:
		raise ValueError('fcoefs parameter passed to ERBFilterBank is the wrong size.')

	if max(np.shape(x)) != np.shape(x)[0]:
		x = np.transpose(x)

	A0 = fcoefs[0]
	A11 = fcoefs[1]
	A12 = fcoefs[2]
	A13 = fcoefs[3]
	A14 = fcoefs[4]
	A2 = fcoefs[5]
	B0 = fcoefs[6]
	B1 = fcoefs[7]
	B2 = fcoefs[8]
	gain = fcoefs[9]

	output = np.zeros((len(gain), max(np.shape(x))))
	for chan in range(len(gain)):
		y1=signal.lfilter([A0[chan]/gain[chan], A11[chan]/gain[chan], A2[chan]/gain[chan]],[B0[chan], B1[chan], B2[chan]], x)
		y2=signal.lfilter([A0[chan], A12[chan], A2[chan]],[B0[chan], B1[chan], B2[chan]], y1)
		y3=signal.lfilter([A0[chan], A13[chan], A2[chan]],[B0[chan], B1[chan], B2[chan]], y2)
		y4=signal.lfilter([A0[chan], A14[chan], A2[chan]],[B0[chan], B1[chan], B2[chan]], y3)
		output[chan, :] = y4

	#if False:
		#semilogx((0:(length(x)-1))*(fs/length(x)),20*log10(abs(fft(output))))

	return output
