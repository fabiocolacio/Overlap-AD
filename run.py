
#################################################################

# Argument parser

import argparse

parser = argparse.ArgumentParser(description='Perform data anomaly detection with LCE')
parser.add_argument(dest='dataset', type=str, help='Choose: AAPL, GOOG, FB, IBM')
parser.add_argument(dest='sample_size', type=int, help='the size of a training data')
parser.add_argument(dest='m', type=int, help='minimum cluster number')

# parser.add_argument(dest='end_sub_sample', type=int, help='maximum data to observe')
# parser.add_argument(dest='forgiven_index', type=int, help='forgiven_index: (int)')
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
import os
import collections
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import json
fig = plt.figure()



def write_our_anomaly(f, result_timestamp):
	hp.write_transcript(f, '=== Our Anomaly Timestamps ===')
	for ts in result_timestamp:
		hp.write_transcript(f, ts)

def write_ground_truth_anomaly_windows(f, filepath):
	hp.write_transcript(f, '=== Ground Truth Anomaly Windows ===')
	hp.write_labeled_json(f,
						'./labels/combined_windows.json',
						filepath)

def write_ground_truth_anomaly(f, filepath):
	hp.write_transcript(f, '= Ground Truth Anomaly Timestamp ===')
	ground_truth_timestamps =  hp.write_labeled_json(f,
								'./labels/combined_labels.json',
								filepath)
	ground_truth_datastamp_list = [-100] * (len(datastamp) - 1)
	for i in range(1, len(ground_truth_datastamp_list)):
		if(dims[1][i] in ground_truth_timestamps):
			ground_truth_datastamp_list[i] = max(dims[2][0:cap_data+2]) / 2
	return ground_truth_datastamp_list

def graph_data(result_list, ground_truth_datastamp_list):

	result, = plt.plot(datastamp[1:cap_data], result_list, '^', markersize=np.sqrt(10.), c='r')
	ground_truth_datastamp, = plt.plot(datastamp[1:cap_data], ground_truth_datastamp_list, 'v', markersize=np.sqrt(10.), c='y')

	origin, = plt.plot(datastamp[1:cap_data], dims[2][1:cap_data], 'o', markersize=np.sqrt(10.), c='b')

	plt.title('Anomaly Detection on NAB on '+ str(args.dataset))
	plt.ylabel('Tweet Numbers')
	plt.xlabel('Timestamp')
	plt.ylim(bottom=0)
	plt.legend([result, ground_truth_datastamp, origin], ['Anomaly', 'Ground Truth', 'Data'])
	dm.mkdir('results')
	fig.savefig('./results/'+str(args.dataset)+"_"+str(args.sample_size)+"_stables_"+str(cap_data), dpi=300)
	plt.clf()


if __name__ == "__main__":
	matplot_color = ['b','g','r','c','m','y','k','w']
	init_data = args.sample_size
	m = args.m
	benchmarks = 15
	result_list = []
	result_timestamp = []
	dims = []

	data = pd.read_csv('./nab/realTweets/realTweets/Twitter_volume_'+str(args.dataset)+'.csv')
	timestamp = np.array(data['timestamp'])
	value = np.array(data['value'])
	datastamp = list(range(0, len(timestamp)))

	# dims[0] = datastamp, dims[1] = timestamp, dims[2] = value
	dims.append(datastamp)
	dims.append(timestamp)
	dims.append(value)

	cap_data = len(datastamp)

	init_cap = init_data - 1 # max data to be computed

	mid_graph = max(dims[2][0:cap_data+2]) / 2 # For plotting anomaly and ground truth

	trusted_data, all_data = hp.deque_list(dims[2], init_data)
	
	# Initially define a number of trusted data to the result_list.
	for k in range(0, len(trusted_data)):
		result_list.append('N')

	# Removing duplicates of data and initializing min and max of the window
	trusted_set = list(dict.fromkeys(trusted_data))
	trusted_set_len = len(trusted_set)
	all_min = min(trusted_set)
	all_max = max(trusted_set)

	# Scale the current data 
	trusted_scaled_data = scaler.scale_data(trusted_set, all_min, all_max)
	
	# Compute the initial LCE
	trusted_LCE = mi.Cluster_Sum(trusted_scaled_data, m, 10)
	index_LCE, val_LCE = hp.compute_LCE_index_val(trusted_LCE)
	
	j = 0
	while(init_data < cap_data-1):
		print('Data stamp: ', init_data)

		obs_entry, all_data = hp.deque_list(all_data, 1)
		obs_data = trusted_set + obs_entry
		all_min = min(obs_data)
		all_max = max(obs_data)

		obs_set = list(dict.fromkeys(obs_data))
		obs_set_len = len(obs_set)

		prev_state = result_list[-1]

		# Classification #
		if(trusted_set_len == obs_set_len): # If the observing data is a duplicate
			if(prev_state == 'P'):
				result_list[-1] = 'N'
				trusted_data.pop(-1)
			result_list.append('N')
			trusted_data.pop(0)
			trusted_data.append(obs_entry[0])
			print(trusted_data, 'N len')


		else: # else the observing data is not a duplicate
			obs_scaled_data, trusted_scaled_data = scaler.scale_data(obs_set, all_min, all_max), scaler.scale_data(trusted_set, all_min, all_max)
			
			trusted_LCE = mi.Cluster_Sum(trusted_scaled_data, m, 10)
			obs_LCE = mi.Cluster_Sum(obs_scaled_data, m, 10)
			
			trusted_index_LCE = hp.get_LCE_index(trusted_LCE)

			trusted_val_LCE = hp.get_LCE_val(trusted_LCE, trusted_index_LCE)
			obs_val_LCE = hp.get_LCE_val(obs_LCE, trusted_index_LCE)

			if(prev_state == 'N' and (obs_val_LCE > trusted_val_LCE)): # N -> N N
				result_list[-1] = 'N'
				result_list.append('N')
				trusted_data.pop(0)
				trusted_data.append(obs_entry[0])
				print(trusted_data, 'N >')

			elif(prev_state == 'N' and (obs_val_LCE == trusted_val_LCE)): # N -> N P
				result_list[-1] = 'N'
				result_list.append('P')

				trusted_data.append(obs_entry[0])
				print(trusted_data, 'N ==')

			elif(prev_state == 'P' and (obs_val_LCE > trusted_val_LCE)): # P -> N N
				# New node landed in same cell as prev node
				result_list[-1] = 'N'
				result_list.append('N')
				trusted_data.pop(0)
				trusted_data.pop(0)
				trusted_data.append(obs_entry[0])
				print(trusted_data, 'P >')


			elif(prev_state == 'P' and (obs_val_LCE == trusted_val_LCE)): # P -> A P
				result_list[-1] = 'A'
				result_list.append('P')
				trusted_data.pop(-1)
				trusted_data.append(obs_entry[0])
				print(trusted_data, 'P ==')


			elif(prev_state == 'A' and (obs_val_LCE > trusted_val_LCE)): # A -> A N
				result_list[-1] = 'A'
				result_list.append('N')
				trusted_data.pop(0)
				trusted_data.append(obs_entry[0])
				print(trusted_data, 'A >')


			elif(prev_state == 'A' and (obs_val_LCE == trusted_val_LCE)): # A -> A P
				result_list[-1] = 'A'
				result_list.append('P')
				trusted_data.pop(0)
				trusted_data.append(obs_entry[0])
				print(trusted_data, 'A ==')

			else: # This is error classification and should not be in the result
				result_list.append('E')
				print('ERROR')


		print()
		print()
		init_data += 1


	# Sum up the results
	n, a, p, e, null = 0, 0, 0, 0, 0
	for i in range(0,len(result_list)):
		if(result_list[i] == 'N'):
			n += 1
			result_list[i] = -100
		elif(result_list[i] == 'A'):
			a += 1
			result_list[i] = mid_graph
			result_timestamp.append(dims[1][i])
		elif(result_list[i] == 'P'):
			p += 1
			result_list[i] = mid_graph - 200
		elif(result_list[i] == 'E'):
			e += 1
			result_list[i] = mid_graph - 600
		else:
			null += 1
			result_list[i] = -100

	print('A: ', a)
	print('N: ', n)
	print('P: ', p)
	print('E: ', e)
	print('null: ', null)
	
	f = open("./results/transcript_"+str(args.dataset)+"_"+str(args.sample_size)+"_stables_"+str(cap_data)+".txt", "w")
	# Processing transcript of data
	ground_truth_datastamp_list = write_ground_truth_anomaly(f, 'realTweets/Twitter_volume_'+str(args.dataset)+'.csv')
	write_ground_truth_anomaly_windows(f, 'realTweets/Twitter_volume_'+str(args.dataset)+'.csv')
	write_our_anomaly(f, result_timestamp)

	# Graph the result
	graph_data(result_list, ground_truth_datastamp_list)

	tp, tn, fp, fn, ud = 0, 0, 0, 0, 0

	for i in range(0,len(datastamp)-1):

		result, ground_truth = result_list[i], ground_truth_datastamp_list[i]

		if(result == ground_truth): # Positive
			if(result < 0 and ground_truth < 0): # Both normal
				tn += 1
			elif(result > 0 and ground_truth > 0): # Both anomaly
				tp += 1
			else:
				ud += 1
		else: # Negative
			if(result > ground_truth): # Our result flagged anomaly on a normal data
				fp += 1
			elif(result < ground_truth): # Our result did not flag anomaly on an anomaly data
				fn += 1
			else:
				ud += 1


	print()
	f.write(hp.output('=== Summary ==='))
	f.write(hp.output('True Negative  :  '+ str(tn)))
	f.write(hp.output('True Positive  :  '+ str(tp)))
	f.write(hp.output('False Negative :  '+ str(fn)))
	f.write(hp.output('False Positive :  '+ str(fp)))
	f.write(hp.output('Total Data: '+ str(len(result_list))))
	print()
	f.close()


