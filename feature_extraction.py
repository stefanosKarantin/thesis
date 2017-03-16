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

badfilenames=[]

pathname = 'aibo-data-train/'
for filename in os.listdir(pathname):
	if filename.endswith('.wav') or filename.endswith('.mp3'):
		if fnmatch.fnmatch(filename,'*A.*'):
			an = get_feature_table(pathname+filename)
			if an is not None:
				anger.append(an)
			else:
				badfilenames.append(filename)
		elif fnmatch.fnmatch(filename,'*E*'):
			e = get_feature_table(pathname+filename)
			if e is not None:
				emphatic.append(e)
			else:
				badfilenames.append(filename)
		elif fnmatch.fnmatch(filename,'*N*'):
			n = get_feature_table(pathname+filename)
			if n is not None:
				neutral.append(n)
			else:
				badfilenames.append(filename)
		elif fnmatch.fnmatch(filename,'*M*'):
			m = get_feature_table(pathname+filename)
			if m is not None:
				motherese.append(m)
			else:
				badfilenames.append(filename)

training_set = {'anger': anger, 'emphatic': emphatic, 'neutral': neutral, 'motherese': motherese}

#print badfilenames
print anger[45].shape
print emphatic[100].shape
