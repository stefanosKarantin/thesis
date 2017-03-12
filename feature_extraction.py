import sys
sys.path.append('./pymir')

import os
import fnmatch

import numpy
from get_features import get_feature_table

anger = []
emphatic = []
neutral = []
motherese = []

pathname = 'aibo-data-train/'
for filename in os.listdir(pathname):
	if filename.endswith('.wav') or filename.endswith('.mp3'):
		if fnmatch.fnmatch(filename,'*A*'):
			anger.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*E*'):
			emphatic.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*N*'):
			neutral.append(get_feature_table(pathname+filename))
		elif fnmatch.fnmatch(filename,'*M*'):
			motherese.append(get_feature_table(pathname+filename))

training_set = {'anger': anger, 'emphatic': emphatic, 'neutral': neutral, 'motherese': motherese}

print anger
