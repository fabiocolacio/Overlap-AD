
#################################################################

# Argument parser

import argparse

parser = argparse.ArgumentParser(description='Perform data anomaly detection with LCE')
parser.add_argument(dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
parser.add_argument(dest='sample_size', type=int, help='the size of a training data')
# parser.add_argument(dest='end_sub_sample', type=int, help='maximum data to observe')
parser.add_argument(dest='forgiven_index', type=int, help='forgiven_index: (int)')
args = parser.parse_args()

################################################################

import sys
import matplotlib
import numpy as np
import pandas as pd
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
import json
fig = plt.figure()


if __name__ == "__main__":
	# Parsing argument(s) to variable(s) #
	b2 = args.sample_size
	# cap_data = args.end_sub_sample
	forgiven_index = args.forgiven_index

	MI_array = []
	dims = []
	matplot_color = ['b','g','r','c','m','y','k','w']
	b1 = 0
	m = 2
	i = 1
	N = 0
	A = 0
	benchmarks = 15
	result_list = []
	result_timestamp = []


	''' '''

	# dataset = hp.csv_extraction('./nab/realTweets/realTweets/Twitter_volume_'+str(args.dataset)+'.csv',1)

	data = pd.read_csv('./nab/realTweets/realTweets/Twitter_volume_'+str(args.dataset)+'.csv')
	timestamp = np.array(data['timestamp'])
	value = np.array(data['value'])
	datastamp = list(range(0, len(timestamp)))
	cap_data = len(datastamp) - 2
	

	# dims[0] = datastamp, dims[1] = timestamp, dims[2] = value
	dims.append(datastamp)
	dims.append(timestamp)
	dims.append(value)


	# Initialize the range of interval to observe data.
	#	For example, if b2 = 100, we start observe b2+1 because
	#	b2 has a stable data we trust for cluster. cap_data
	#	represents the cap of data we want to examine.
	# total_timestamp = list(range(b2+1, cap_data+1))
	mid_graph = max(dims[2][b2+1:cap_data+2]) / 2

	t0_size = [0, b2]
	t1_size = [0, b2 + 1]
	t2_size = [0, b2 + 2]

	while(i < cap_data - b2 + 1):
		print('\n\n\n============= Timestamp Number: ', str(i+b2), ' =============')
		# Definted window's sizes
		t0_wz = [b1+i, b2+i]
		t1_wz = [b1+i, b2+i +1]
		t2_wz = [b1+i, b2+i +2]

		# Get each window into 3 instances
		t0_sub_dim = mi.get_sub_dims(dims, t0_wz)
		t1_sub_dim = mi.get_sub_dims(dims, t1_wz)
		t2_sub_dim = mi.get_sub_dims(dims, t2_wz)

		# print(t0_sub_dim)

		# Pre-define the scale for this current iteration 
		# We take max and min from all 3 windows at this iteration because
		# 	we do not want to shift around the data within the interval.
		#	Remember, LCE is purely based on data stabality on the dimension,
		#	thus, NO data should be rescaled or move when comparison between
		#	windows happen.
		all_min = min(min(t0_sub_dim[2]), min(t1_sub_dim[2]), min(t2_sub_dim[2]))
		all_max = max(max(t0_sub_dim[2]), max(t1_sub_dim[2]), max(t2_sub_dim[2]))


		# # Scale 3 windows based on min and max of their 3 windows.
		scaled_t0 = scaler.scale_data(t0_sub_dim[2], all_min, all_max)
		scaled_t1 = scaler.scale_data(t1_sub_dim[2], all_min, all_max)
		scaled_t2 = scaler.scale_data(t2_sub_dim[2], all_min, all_max)

		# print('Max observe data : ', observe_2_dims)
		# print('Scaled t0      : ', scaled_t0)
		# print('Scaled t1      : ', scaled_t1)
		# print('Scaled t2      : ', scaled_t2)

		# print('All min: ', all_min)
		# print('All max: ', all_max)

		# print()

		# # Extra, no need to worry about this for now
		# MI_main = mi.MINDID_Multi_Dims([scaled_main], m, 15, main_bucket_size)
		# MI_obs_1 = mi.MINDID_Multi_Dims([scaled_obs_1], m, 15, observe_1_bucket_size)
		# MI_obs_2 = mi.MINDID_Multi_Dims([scaled_obs_2], m, 15, observe_2_bucket_size)

		# # Removing duplicates so that each precise number is unique and does not
		# #	cause infinite LCE value
		scaled_set_t0 = [list(dict.fromkeys(scaled_t0))]
		scaled_set_t1 = [list(dict.fromkeys(scaled_t1))]
		scaled_set_t2 = [list(dict.fromkeys(scaled_t2))]

		# print('Non dupe Main      : ', scaled_set_t0)
		# print('Non dupe Observe 1 : ', scaled_set_t1)
		# print('Non dupe Observe 2 : ', scaled_set_t2)

		# # Define bucket sizes after duplicates have been removed
		t0_set_wz = [0, len(scaled_set_t0[0])]
		t1_set_wz = [0, len(scaled_set_t1[0])]
		t2_set_wz = [0, len(scaled_set_t2[0])]

		
		# # Compute the score of LCE, this will result in list of cluster sum
		CS_t0_LCE = mi.Cluster_Sum(scaled_set_t0, m, 10, t0_set_wz)
		CS_t1_LCE = mi.Cluster_Sum(scaled_set_t1, m, 10, t1_set_wz)
		CS_t2_LCE = mi.Cluster_Sum(scaled_set_t2, m, 10, t2_set_wz)

		# # Get the LCE index
		t0_LCE = hp.compute_LCE_index(CS_t0_LCE)
		t1_LCE = hp.compute_LCE_index(CS_t1_LCE)
		t2_LCE = hp.compute_LCE_index(CS_t2_LCE)


		# print('#######################\n')


		# print('Length t0: ', len(scaled_set_t0[0]))
		# print('Length t1: ', len(scaled_set_t1[0]))
		# print('Length t2: ', len(scaled_set_t2[0]))

		# print('t0 LCE: ',CS_main_LCE[0])
		# print('t1 LCE: ',CS_obs_1_LCE[0])
		# print('t2 LCE: ',CS_obs_2_LCE[0])

		# print()
		# print('#######################\n')


		# Start codition
		if( (len(scaled_set_t0[0]) == len(scaled_set_t1[0]) or len(scaled_set_t1[0]) == len(scaled_set_t2[0])) or
			(CS_t0_LCE[0][t0_LCE - forgiven_index] < CS_t1_LCE[0][t1_LCE - forgiven_index] or CS_t1_LCE[0][t1_LCE - forgiven_index] < CS_t2_LCE[0][t2_LCE - forgiven_index])):
			result_list.append(-50)

		else:
			result_list.append(mid_graph)
			result_timestamp.append(dims[1][i])

		# print('Result', result_list)
		i += 1

	for d in result_list:
		if(d == -50):
			N += 1
		else:
			A += 1

	print()
	print("=== Total classification ===")
	print('Normal: ', N)
	print('Anomaly: ', A)
	print()



	f = open("./transcript_"+str(args.dataset)+"_"+str(b2)+"_stables_"+str(cap_data)+"_forgivenIndex_"+str(forgiven_index)+".txt", "w")

	ground_truth_timestamps = []

	f.write(hp.output('=== Ground Truth Anomaly Timestamp ==='))

	with open('./labels/combined_labels.json') as json_file:
		data = json.load(json_file)
		for windows in data['realTweets/Twitter_volume_'+str(args.dataset)+'.csv']:
 			f.write(hp.output(windows))
 			ground_truth_timestamps.append(windows)
	print()

	ground_truth_datastamp_list = [-50] * len(datastamp)

	for i in range(1, len(ground_truth_datastamp_list)):
		if(dims[1][i] in ground_truth_timestamps):
			ground_truth_datastamp_list[i] = max(dims[2][b2+1:cap_data+2]) / 2

	f.write(hp.output('=== Ground Truth Anomaly Windows ==='))
	with open('./labels/combined_windows.json') as json_file:
		data = json.load(json_file)
		for windows in data['realTweets/Twitter_volume_'+str(args.dataset)+'.csv']:
			f.write(hp.output(str(windows)))
	print()

	f.write(hp.output('=== Our Anomaly Timestamps ==='))
	for ts in result_timestamp:
		f.write(hp.output(ts))
	print()

	# Plot data
	result, = plt.plot(datastamp[b2+1:cap_data+1], result_list, '^', markersize=np.sqrt(10.), c='r')
	# plt.clf()
	ground_truth_datastamp, = plt.plot(datastamp[0:len(datastamp)], ground_truth_datastamp_list, 'v', markersize=np.sqrt(10.), c='m')


	# # Plot actual data
	origin, = plt.plot(datastamp[b2+1:cap_data+1], dims[2][b2:cap_data], 'o', markersize=np.sqrt(10.), c='b')
	plt.title('Anomaly Detection on NAB on '+ str(args.dataset))
	plt.ylabel('Tweet Numbers')
	plt.xlabel('Timestamp')
	plt.ylim(bottom=0)
	plt.legend([result, ground_truth_datastamp, origin], ['Anomaly', 'Ground Truth', 'Data'])
	dm.mkdir('results')
	fig.savefig('./results/'+str(args.dataset)+"_"+str(b2)+"_stables_"+str(cap_data)+"_forgivenIndex_"+str(forgiven_index), dpi=300)
	plt.clf()
	f.close()