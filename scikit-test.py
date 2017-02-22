import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
digits = load_digits()
data = scale(digits.data)

print len(digits['target'])
print len(data[0])