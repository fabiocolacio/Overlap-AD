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