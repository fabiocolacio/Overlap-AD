
#################################################################

# Argument parser

import argparse

parser = argparse.ArgumentParser(description='Perform data anomaly detection with LCE')
parser.add_argument(dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
parser.add_argument(dest='sample_size', type=int, help='the size of a training data')
parser.add_argument(dest='end_sub_sample', type=int, help='maximum data to observe')
args = parser.parse_args()

################################################################

import sys
import matplotlib
import numpy as np
import lib.helper as hp
import lib.scaler as scaler
import lib.dir_manager as dm
import lib.morisita_index as mi
import lib.grapher
import lib.training
import os
import collections
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
fig = plt.figure()

# Parsing argument(s) to variable(s) #
b2 = args.sample_size
cap_data = args.end_sub_sample


MI_array = []
LNMI = []
dims = []
matplot_color = ['b','g','r','c','m','y','k','w']
b1 = 0
m = 2
i = 1
benchmarks = 15
result_list = []

''' '''

dataset = hp.csv_extraction('./nab/realTweets/realTweets/Twitter_volume_'+str(args.dataset)+'.csv',1)

timestamp = hp.get_col(dataset, 0)
tweetCount = hp.get_col(dataset, 1)

arr_month, arr_date, arr_time, arr_tweet = hp.disect_dimensions(timestamp, tweetCount, 0, cap_data+2) # cluster_x is timestamp, cluster_y is tweets number

arr_tweet = list(map(int, arr_tweet))
dims.append(arr_tweet)


# Initialize the range of interval to observe data.
#	For example, if b2 = 100, we start observe b2+1 because
#	b2 has a stable data we trust for cluster. cap_data
#	represents the cap of data we want to examine.
total_timestamp = list(range(b2+1, cap_data+1))
mid_graph = max(dims[0][b2+1:cap_data+2]) / 2

main_bucket_size = [0, b2]
observe_1_bucket_size = [0, b2 + 1]
observe_2_bucket_size = [0, b2 + 2]

while(i < cap_data - b2 + 1):
	print('\n\n\n============= Timestamp Number: ', str(i+b2), ' =============')
	# Definted window's sizes
	main_bucket = [b1+i, b2+i]
	observe_1_bucket = [b1+i, b2+i +1]
	observe_2_bucket = [b1+i, b2+i +2]

	# Get each window into 3 instances
	main_dims = mi.get_sub_dims(dims, main_bucket)
	observe_1_dims = mi.get_sub_dims(dims, observe_1_bucket)
	observe_2_dims = mi.get_sub_dims(dims, observe_2_bucket)

	# Pre-define the scale for this current iteration 
	# We take max and min from all 3 windows at this iteration because
	# 	we do not want to shift around the data within the interval.
	#	Remember, LCE is purely based on data stabality on the dimension,
	#	thus, NO data should be rescaled or move when comparison between
	#	windows happen.
	all_min = min(min(main_dims[0]), min(observe_1_dims[0]), min(observe_2_dims[0]))
	all_max = max(max(main_dims[0]), max(observe_1_dims[0]), max(observe_2_dims[0]))


	# Scale 3 windows based on min and max of their 3 windows.
	scaled_main = scaler.scale_data(main_dims[0], all_min, all_max)
	scaled_obs_1 = scaler.scale_data(observe_1_dims[0], all_min, all_max)
	scaled_obs_2 = scaler.scale_data(observe_2_dims[0], all_min, all_max)

	print('Max observe data : ', observe_2_dims)
	# print('Scaled Main      : ', scaled_main)
	# print('Scaled Obs 1     : ', scaled_obs_1)
	# print('Scaled Obs 2     : ', scaled_obs_2)

	print('All min: ', all_min)
	print('All max: ', all_max)

	print()

	# Extra, no need to worry about this for now
	MI_main = mi.MINDID_Multi_Dims([scaled_main], m, benchmarks, main_bucket_size)
	MI_obs_1 = mi.MINDID_Multi_Dims([scaled_obs_1], m, benchmarks, observe_1_bucket_size)
	MI_obs_2 = mi.MINDID_Multi_Dims([scaled_obs_2], m, benchmarks, observe_2_bucket_size)

	# Define bucket sizes after duplicates have been removed
	main_bucket_size = [0, len(scaled_main)]
	observe_1_bucket_size = [0, len(scaled_obs_1)]
	observe_2_bucket_size = [0, len(scaled_obs_2)]


	# print('MI_t0: ', CS_main_LCE[0])
	# print('MI_t1: ', CS_obs_1_LCE[0])
	# print('MI_t2: ', CS_obs_2_LCE[0])


	print('#######################\n')


	print('MI Main: ', MI_main[0])
	print('MI Obs 1: ', MI_obs_1[0])
	print('MI Obs 2: ', MI_obs_2[0])

	print()
	print('#######################\n')

	MI_1, = plt.plot(MI_main[1], MI_main[0], 'o', markersize=np.sqrt(10.), c='g')
	MI_2, = plt.plot(MI_obs_1[1], MI_obs_1[0],'o', markersize=np.sqrt(10.), c='y')
	MI_3, = plt.plot(MI_obs_1[1], MI_obs_2[0],'o', markersize=np.sqrt(10.), c='r')

	LNMI.append(hp.get_last_normal_MI(MI_main[0],MI_obs_1[0],MI_obs_2[0]))

	plt.title('Anomaly Detection on NAB on '+ str(args.dataset))
	plt.ylabel('MI Value')
	plt.xlabel('Delta')
	plt.ylim(1, 10)
	plt.legend([MI_1, MI_2, MI_3], ['MI_1', 'MI_2', 'MI_3'])
	dm.mkdir('results/MI_collection')
	fig.savefig('./results/MI_collection/'+str(args.dataset)+"_data_"+str(i), dpi=500)
	plt.clf()


	i += 1

print('->', LNMI)