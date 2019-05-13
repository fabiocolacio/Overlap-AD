from lib.dir_manager import *
from lib.scaler import *
import lib.morisita_index as mi
import lib.grapher
import csv
import datetime
import time
import json
from datetime import timezone
import numpy as np
from sklearn.decomposition import PCA
import math
import os


'''
	Get min and max of a 2D matrix
	Param: 2D matrix
	Return: 2 arguments, 1st arg = max array, 2nd arg = min array
'''
def find_min_max(data):
	''' Convert to numpy to check dimension '''
	dim = np.array(data)
	max_matrix = []
	min_matrix = []
	if(len(dim.shape) == 1):
		for i in range(0,len(dim.shape)):
			max_matrix.append(data[i])
			min_matrix.append(data[i])
		for row in data:
			max_matrix[0] = max(max_matrix[0], row)
			min_matrix[0] = min(min_matrix[0], row)
	elif(len(dim.shape) == 2):
		for i in range(0,len(data[0])):
			max_matrix.append(data[0][i])
			min_matrix.append(data[0][i])
		for row in data:
			#print(row)
			max_matrix[0] = max(max_matrix[0], row[0])
			max_matrix[1] = max(max_matrix[1], row[1])
			min_matrix[0] = min(min_matrix[0], row[0])
			min_matrix[1] = min(min_matrix[1], row[1])
	
	return max_matrix, min_matrix


'''
	Get key and value of the matrix (x and y)
	Param: 2D matrix
	Return: 2 separate vectors of the data
'''
def get_key_value(data):
	key, val = [], []
	for k in data:
		key.append(k)
		val.append(data[k])
	return key, val


'''
	Convert 2D dataset from string to float type
    Param: 	2D matrix
	Return:	2D matrix (in float)
'''
def matrix_float_conversion(NxE):
	ret_matrix = []
	if(len(NxE[0]) != 2):
		print('ERROR: Not 2D dataset!')
		return 0
	for row in NxE:
		ret_matrix.append( [ float(row[0]), float(row[1]) ] )
	return ret_matrix



def compute_MI_peek(MI_arr):
	peek_MI = 0
	peek_delta = 0
	peek_benchmark = 0
	for i in range(0, len(MI_arr[0])):
		if(MI_arr[1][i] > peek_MI):
			peek_MI = MI_arr[1][i]
			peek_delta = MI_arr[0][i]
			peek_benchmark = i
	return peek_MI, peek_delta, peek_benchmark

def compute_LCE_index_val(LCE):
	LCE_benchmark = 0
	for i in range(0, len(LCE[0])):
		if(LCE[0][i] > 0):
			LCE_benchmark = i
			LCE_val = LCE[0][i]
	return LCE_benchmark, LCE_val

def get_LCE_index(LCE):
	LCE_benchmark = 0
	for i in range(0, len(LCE[0])):
		if(LCE[0][i] > 0):
			LCE_benchmark = i
	return LCE_benchmark

def get_LCE_val(LCE, LCE_benchmark):
	return LCE[0][LCE_benchmark]


def get_last_normal_MI(MI_0, MI_1, MI_2):
	index = 1
	inverse = False
	for i in range(1, len(MI_0)):
		if(MI_0[i] < MI_1[i] and MI_1[i] < MI_2[i]):
			index += 1
			inverse = True
		elif(inverse):
			return index
		else:
			pass
	return -1

def pending_state(t1, t2):

	return 0

def check_grey_cell(result_list, t2_anomaly):
	if(result_list[-1] == 0 and t2_anomaly): # Grey cell

		pass
	elif(result_list[-1] == -1): # Black cell
		pass
	elif(result_list[-1] == 1): # White cell
		pass
	else: # Invalid
		pass

def deque_list(data, num_data):
	ret_pop_data = list(data[:num_data])
	ret_origin_data = list(data[num_data:])
	return ret_pop_data, ret_origin_data

def output(str):
	print(str)
	return str+'\n'



def write_transcript(f, text):
	f.write(output(text))


def write_labeled_json(f, file_path, dataset_path):
	ret_arr_timestamp = []
	with open(file_path) as json_file:
		data = json.load(json_file)
		for windows in data[dataset_path]:
			f.write(output(str(windows)))
			ret_arr_timestamp.append(windows)
	return ret_arr_timestamp