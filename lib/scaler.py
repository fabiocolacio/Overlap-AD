from lib.dir_manager import *
import csv
import datetime
import time
from datetime import timezone
import numpy as np
from sklearn.decomposition import PCA
import math


'''
	Convert the timestamp string and return seconds of the day
    Param:	1D array, 1D array
	Return:	4 parameters of data
'''
def get_second(timestamp):
	h, m, s = timestamp.split(':')
	return (int(h) * 3600) + (int(m) * 60) + (int(s))

'''
	Convert the timestamp string and return seconds of the day
    Param:	1D array, 1D array
	Return:	4 parameters of data
'''
def scale_data(x, low, high):
	_min = float(low)
	_max = float(high)
	ret_arr = []
	for i in x:
		ret_arr.append( (float(i) - _min) / (_max - _min) )
	return ret_arr