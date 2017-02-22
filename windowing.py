import sys
sys.path.append('./pymir')

import decimal
import math
import numpy as np
import scipy.io.wavfile as wav
from pymir import Frame

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def windowing(sig,fs,winlen=0.025,winhop=0.01,window=None):

	length = len(sig) 
	overlap = int(fs*winhop)
	framesize = int(winlen*fs)
	window = window(framesize)
	number_of_frames = (length/overlap)

	#length of DFT ,change here
	nfft_length = framesize 
	#print number_of_frames
	frametype = sig.dtype
	# This declares a 2D matrix,with rows equal to the number of frames,and columns equal to the framesize or the length of each DTF
	frames = Frame.Frame(frametype,(number_of_frames,framesize)) 
	for k in range(0,number_of_frames):
		for i in range(0,framesize):
			if((k*overlap+i) < length):
				frames[k][i] = sig[k*overlap+i]*window[i]
			else:
				frames[k][i] = 0

	return frames

def windowing2(sig,fs,winlen=0.025,winhop=0.01,window=lambda x:np.ones((x,))):
    slen = len(sig)
    #print slen
    winlen = int(round_half_up(fs*winlen))
    #print frame_len
    winhop = int(round_half_up(fs*winhop))
    #print frame_step
    if slen <= winlen:
    	numframes = 1
    else:
    	numframes = 1 + int(math.ceil((1.0*slen - winlen)/winhop))

    padlen = int((numframes-1)*winhop + winlen)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig,zeros))

    indices = np.tile(np.arange(0,winlen),(numframes,1)) + np.tile(np.arange(0,numframes*winhop,winhop),(winlen,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(window(winlen),(numframes,1))

    myframes = frames*win
    return myframes