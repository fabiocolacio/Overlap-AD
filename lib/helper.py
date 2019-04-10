from lib.dir_manager import *
from lib.scaler import *
import lib.morisita_index as mi
import lib.grapher
import csv
import datetime
import time
from datetime import timezone
import numpy as np
from sklearn.decomposition import PCA
import math
import os
'''
	Start pre-processing data
	Param: path to the file in nab
	Return: 2D matrix of a data (1st col unix time, 2nd col val)
'''

def start_process(file_path):	
	# row[0] => date and time
	# row[1] => mentioned
	# Using UNIX timestamp from January 1, 1970
	column_num = 0
	row_num = 0
	 
	with open(file_path) as csvfile:
		spamreader = iter(csv.reader(csvfile, delimiter=','))
		next(spamreader)
		ret_matrix = []
		for row in spamreader:
			# Check for columns
			if column_num == 0:
				column_num = len(row)

			#yyyy-mm-dd hh:mm:ss			
			''' print raw data from csv '''
			#print('x: ', row[0])
			#print('y: ', row[1])
			year = int(row[0][0:4])
			month = int(row[0][5:7])
			day = int(row[0][8:10])
			hour = int(row[0][11:13])
			minute = int(row[0][14:16])
			second = int(row[0][17:22])
			mentions = int(row[1])
			# Print format
			# date time = (year, month, date, hour, minute, second)
			dt = datetime.datetime(year, month, day, hour, minute, second)
			unix_time = time.mktime(dt.timetuple())
			ret_matrix.append([unix_time, mentions])
			''' print unix time  '''
			#print(unix_time)
			row_num += 1
		print('--------------------------------')
		print('Rows Number: ', row_num)
		print('Columns Number: ', column_num)
		print('--------------------------------')
		return ret_matrix

'''
	Show the param data
	Param: 2D matrix
	Return: None (print data)
'''
def show_data(data):
	for i in data:
		print(i)	

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
	Get a column value of the 2D matrix
	Param: data = 2D matrix, index = number of column (0 for first col)
	Return: An array of specified column
'''
def get_col(data, index):
	ret_col = []
	for i in range(0,len(data)):
		ret_col.append(data[i][index])
	return ret_col
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
	Start pre-processing data
	Param: path to the file in nab
	Return: 2D matrix of a data (1st col unix time, 2nd col val)
'''
def start_process_freq(file_path):	
	# row[0] => date and time
	# row[1] => mentioned
	# Using UNIX timestamp from January 1, 1970
	column_num = 0
	row_num = 0
	freq_dict = dict()
	 
	with open(file_path) as csvfile:
		spamreader = iter(csv.reader(csvfile, delimiter=','))
		next(spamreader)
		ret_matrix = []
		for row in spamreader:
			# Check for columns
			if column_num == 0:
				column_num = len(row)
			mentions = int(row[1])
			if row[1] in freq_dict:
				freq_dict[row[1]] += 1
			else:
				freq_dict[row[1]] = 1
			''' print unix time  '''
			#print(unix_time)
			row_num += 1
		for k in freq_dict:
			ret_matrix.append([k, freq_dict[k]])
		print('--------------------------------')
		print('Rows Number: ', row_num)
		print('Columns Number: ', column_num)
		print('--------------------------------')
		return ret_matrix

'''
	Extract data from CSV file, must be 2 columns (2D)
    Param: 	csv_file_path -> Path to CSV file
			skip -> Number of rows to be skipped
	Return:	2D matrix extracted from CSV
'''
def csv_extraction(csv_file_path, skip):	
	# row[0] => first column
	# row[1] => second column
	column_num = 0
	row_num = 0

	with open(csv_file_path) as csvfile:
		ret_matrix = []
		spamreader = iter(csv.reader(csvfile, delimiter=','))
		for i in range(0, skip):
			next(spamreader)
		for row in spamreader:
			# Check for columns
			if column_num == 0:
				column_num = len(row)

			row1 = row[0]
			row2 = row[1]
			ret_matrix.append([row1, row2])
			''' print unix time  '''
			row_num += 1
		print('--------------------------------')
		print('Rows Number: ', row_num)
		print('Columns Number: ', column_num)
		print('--------------------------------')
		return ret_matrix

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


'''
	Append the timestamp with incrementing counter
    Param:	NxE -> 2D dataset
	Return:	3D dataset -> (timestamp, data&time, vals)
		Note: return matrix (int, string, int)
'''
def nab_timestamp_conversion(NxE):
	ret_matrix = []
	if(len(NxE[0]) != 2):
		print('ERROR: Not 2D dataset!')
		return 0
	for i in range(0,len(NxE)):
		ret_matrix.append( [ i, NxE[i][0] , int(NxE[i][1]) ] )
	return ret_matrix
    

'''
	Concat dimension
	Param: Axis x, Axis y
	Return: Matrix of 2D, x as the first dimension and y as the second dimension
'''
def concat_dimension(x, y):
	ret_matrix = []
	if(len(x) != len(y)):
		print('Mismatch array length')
		return None
	for i in range(0,len(x)):
		ret_matrix.append([x[i],y[i]])
	return ret_matrix

'''
	Convert dataset from "http://cs.joensuu.fi/sipu/datasets/" to csv
	Param: path to the text file
	Return: csv file in 2D
'''
def text_to_csv_2D_joensuu(text_file_path):
	mkdir('joensuu_dataset')
	with open(text_file_path) as f:
		with open('joensuu_dataset/s1.csv', 'w') as csv_out:
			spamwriter = csv.writer(csv_out)
			for line in f:
				data = line.split()
				spamwriter.writerow(data)

'''
	Create a CSV file from 3D timestamp to 2D timestamp. Disregard date and time.
	Param: 	3D matrix
	Return: csv file in 2D
'''
def create_day_timestamp(NxE):
	if(len(NxE[0]) != 3):
		print('Not 3D dataset!')
		print('Execute nab_timestamp_conversion first')
		return 0
	mkdir('nab_data')
	date = '00'
	cur_day = open('nab_data/day_'+date+'.csv', 'a', newline='')
	for ts in NxE:
		if(date != ts[1][8:10]):
			cur_day.close()
			date = ts[1][8:10]
			cur_day = open('nab_data/day_'+date+'.csv', 'a', newline='')
			writer = csv.writer(cur_day,delimiter=' ', 	quotechar='|', quoting=csv.QUOTE_MINIMAL)
		print(ts[0])	
		data = str(ts[0])+','+str(ts[2])
		writer.writerow( data.split() )

	cur_day.close()



	'''Extract date '''



'''
	Minimize the data into range 0 to 1 via divider of max base 10
    Param:	2D matrix
	Return:	2D matrix (minimized)
'''
def minimizing_matrix(NxE):
	cluster_x = get_col(NxE, 0)
	cluster_y = get_col(NxE, 1)
	ret_matrix = []

	divider = 1;

	for i in range(0,len(cluster_x)):
		if(cluster_x[i] > divider or cluster_y[i] > divider):
			divider *= 10

	for i in range(0,len(cluster_x)):
		ret_matrix.append([cluster_x[i] / divider , cluster_y[i] / divider])


	return ret_matrix

'''
	Convert timestamp into month, date, and time separately
    Param:	1D array, 1D array
	Return:	4 parameters of data
'''
def disect_dimensions(timestamp, tweets, lowerBound, upperBound):
	ret_month, ret_date, ret_time, ret_tweets = [], [], [], []
	for i in range(lowerBound, upperBound):
		ret_month.append(timestamp[i][5:7])
		ret_date.append(timestamp[i][8:10])
		ret_time.append(get_second(timestamp[i][11:19]))
		ret_tweets.append(tweets[i])
	return ret_month, ret_date, ret_time, ret_tweets



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

def compute_LCE_index(MI_arr):
	LCE_benchmark = 0
	for i in range(0, len(MI_arr[0])):
		if(MI_arr[0][i] > 0):
			LCE_benchmark = i
	return LCE_benchmark


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


def start_MI_computation(dims, bucket_range, benchmarks):
	# print('########### SAMPLE 1 #############')
	# mkdir('./simple_sample/')

	bucket = 200
	MAX_MI_VAL = 5000
	# benchmarks = 10
	MI_array = []
	# dims = []
	matplot_color = ['b','g','r','c','m','y','k','w']
	m = 2
	# bucket_size = 2
	# bucket_range = [1,10]


	# print_result(f, '########### Dt #############\n')
	grapher.create_skeleton('simple_sample/sample1',len(dims))
	grapher.benchmark_division('simple_sample/sample1', len(dims), [0,0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875])
	grapher.plot_data('simple_sample/sample1', dims, bucket_range)
	# Note: Look at 2nd benchmark, it should be 0
	sub_dims = mi.get_sub_dims(dims, bucket_range)
	MI, delta = mi.MINDID_Multi_Dims(sub_dims, m, benchmarks, bucket_range)
	
	MI_array = [delta, MI]
	# plt.cla()
	# plt.clf()

	grapher.plot_MI('none', MI_array)
	''' Normal MI '''
	# for j in range(0, len(MI_array)):
		# plt.plot(MI_array[j][0], MI_array[j][1], 'bo')
		# plt.plot(MI_array[j][0], MI_array[j][1], 'b')

		# #plt.plot([0,1,2,3,4,5], [1,1,1,1,1,1], 'g')

		# plt.axis([0,1.5,0,MAX_MI_VAL])
		# mkdir('graphs_MI')
		# fig.savefig('graphs_MI/cluster_dt_'+str(j), dpi=500)
		# plt.cla()
		# plt.clf()
		# pass

	return MI_array


def start_MI_computation_Norminator(dims, bucket_range, benchmarks):
	# print('########### SAMPLE 1 #############')
	# mkdir('./simple_sample/')

	bucket = 200
	MAX_MI_VAL = 5000
	# benchmarks = 10
	MI_array = []
	# dims = []
	matplot_color = ['b','g','r','c','m','y','k','w']
	m = 2
	# bucket_size = 2
	# bucket_range = [1,10]


	# print_result(f, '########### Dt #############\n')
	# grapher.create_skeleton('simple_sample/sample1',len(dims))
	# grapher.benchmark_division('simple_sample/sample1', len(dims), [0,0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875])
	# grapher.plot_data('simple_sample/sample1', dims, bucket_range)
	# Note: Look at 2nd benchmark, it should be 0
	sub_dims = mi.get_sub_dims(dims, bucket_range)
	MI, delta = mi.MINDID_Multi_Dims_Norminator(sub_dims, m, benchmarks, bucket_range)
	
	MI_array = [delta, MI]
	

	return MI_array