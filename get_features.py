
import numpy as np

from pymir import LinearPredictiveAnalysis
from pymir import AudioFile
from pymir import Energy
from pymir import Onsets
from pymir import Deltas

from python_speech_features import mfcc
from python_speech_features import logfbank

def get_audio(audiopath):
	(audio, fs) = AudioFile.open(audiopath)
	return audio, fs

def get_mfcc(audio, fs):
	mfcc_feat = mfcc(audio,fs,numcep=12)
	return mfcc_feat

def get_lpcc(audio, fs):
	myframes = audio.frames(fs,window=np.hamming)

	lpc_coefs = []
	error = []

	for frame in myframes:
		
		(coef, err) = LinearPredictiveAnalysis.lpc(frame,order=12)
		lpc_coefs.append(coef)
		error.append(err)

	lpcc_coefs = []
	for i in range(0,len(lpc_coefs)):
		coef = LinearPredictiveAnalysis.lpcc(lpc_coefs[i],error[i],order=12)
		lpcc_coefs.append(coef)

	lpcc_coefs = np.asarray(lpcc_coefs)

	return lpcc_coefs

def get_plpc():
	return 1

def get_feature_table(audiopath):

	(audio,fs) = get_audio(audiopath)

	mfcc_coefs = get_mfcc(audio,fs)
	logbank = logfbank(audio,fs,nfilt=1)

	mfcc_Deltas = Deltas.getDeltas(mfcc_coefs)
	log_Deltas = Deltas.getDeltas(logbank)

	mfcc_DeltaDeltas = Deltas.getDeltas(mfcc_Deltas)
	log_DeltaDeltas = Deltas.getDeltas(log_Deltas)
	
	lpcc_coefs = get_lpcc(audio,fs)
	lpcc_Deltas = Deltas.getDeltas(lpcc_coefs)
	lpcc_DeltaDeltas = Deltas.getDeltas(lpcc_Deltas)

	feature_table = np.concatenate((mfcc_coefs,mfcc_Deltas,mfcc_DeltaDeltas,logbank,log_Deltas,log_DeltaDeltas,lpcc_coefs,lpcc_Deltas,lpcc_DeltaDeltas), axis=1)

	return feature_table
