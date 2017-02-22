import sys
sys.path.append('./pymir')

import os
import fnmatch

import numpy
from get_features import get_feature_table

anger = []
disgust = []
fear = []
happiness = []
neutral = []
pain = []
pleasure = []
sadness = []
surprise = []

pathname = 'emotional_db/'
for filename in os.listdir(pathname):
	if filename.endswith('.wav') or filename.endswith('.mp3'):
		if fnmatch.fnmatch(filename,'*anger*'):
			anger.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*disgust*'):
			disgust.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*fear*'):
			fear.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*happiness*'):
			happiness.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*neutral*'):
			neutral.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*pain*'):
			pain.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*pleasure*'):
			pleasure.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*sadness*'):
			sadness.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*surprise*'):
			surprise.append(get_feature_table(pathname+filename))


training_set = {'anger': anger, 'disgust': disgust, 'fear': fear, 'happiness': happiness, 'neutral': neutral, 'pain': pain, 'pleasure': pleasure, 'sadness': sadness, 'surprise': surprise}

#print training_set
        