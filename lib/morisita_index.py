import math
import sys
import lib.helper

# from dir_manager import *
def MINDID_Multi_Dims(dims, m, benchmarks, bucket_range):

	ret_MI = []
	ret_delta = []
	delta = 1
	L = delta

	dims = get_sub_dims(dims, bucket_range)
	feat_num = len(dims)

	for b_ith in range(1,benchmarks+1):
		Q = compute_Q_Multi_Dims(b_ith, feat_num)
		range_cell = get_range_cell_1D(delta)
		nth_list = get_multi_ith_cell(dims, range_cell, Q, m)
		sum_nth = 0
		cluster_m_sum = nth_summation(nth_list, m)
		N_m_sum = nth_summation([len(dims[0])], m)


	###############
		MI = math.pow(Q, m - 1) * (cluster_m_sum / N_m_sum)
	###############

		my_string = 'MI: '+str(MI)+'\n'
		my_string = 'Q: '+str(Q)+'\n'
		my_string = 'Sum(n): '+str(cluster_m_sum)+'\n'
		my_string = 'Sum(N): '+str(N_m_sum)+'\n'
	
		ret_delta.append(delta)
		ret_MI.append(MI)


		delta = delta / 2


	return ret_MI, ret_delta

def Cluster_Sum(dims, m, benchmarks, bucket_range):

	ret_MI = []
	ret_delta = []
	delta = 1
	L = delta

	dims = get_sub_dims(dims, bucket_range)
	feat_num = len(dims)

	for b_ith in range(1,benchmarks+1):
		Q = compute_Q_Multi_Dims(b_ith, feat_num) # Check
		range_cell = get_range_cell_1D(delta) # Check
		# print('DIM: ',dims)
		nth_list = get_multi_ith_cell(dims, range_cell, Q, m)
		sum_nth = 0
		cluster_m_sum = nth_summation(nth_list, m)
		N_m_sum = nth_summation([len(dims[0])], m)
	
	###############
		MI = cluster_m_sum
	###############

		my_string = 'MI: '+str(MI)+'\n'
		my_string = 'Q: '+str(Q)+'\n'
		my_string = 'Sum(n): '+str(cluster_m_sum)+'\n'
		my_string = 'Sum(N): '+str(N_m_sum)+'\n'

		ret_delta.append(delta)
		ret_MI.append(MI)

		delta = delta / 2

	return ret_MI, ret_delta

# Returns list of overlapping cell
def get_multi_ith_cell(dims, range_cell, Q, m):

	cell_hash = {}
	n_ith_arr = []
	dim_data = []
	feat_ith_data = []
	cell_num = 1
	feature_counter = 1

	for feat in dims:
		for i in feat:
			cell_num = 1
			for r in range_cell:
				if(i >= r[0] and i <= r[1]):
					break # break 'r' loop
				cell_num += 1
			feat_ith_data.append(cell_num)
		dim_data.append(feat_ith_data)
		feat_ith_data = []
		feature_counter += 1


	final_dim = [*zip(*dim_data)] # Transpose the data, flip x to y, and y to x

	for i in range(0, len(final_dim)):
		if final_dim[i] in cell_hash:
			cell_hash[final_dim[i]] += 1
		else:
			cell_hash[final_dim[i]] = 1


	for freq in cell_hash.values():
		if(freq >= m):
			n_ith_arr.append(freq)


	return (n_ith_arr)


'''
	Compute number of quadrats. This is computed by cells number of 1 dimension multiply by number of dimensions
'''
def compute_Q_Multi_Dims(b_ith, feat_num):
	return math.pow(2, ( (b_ith-1) * feat_num))

'''
	Get index of cells from the dimension
'''
def get_index_ith_cell_2D(NxE, range_Q, feat):
	feat -= 1
	ret_ith_cell = []
	for data in NxE:
		for i, intv in enumerate(range_Q):
			if(data[feat] >= intv[0] and data[feat] < intv[1]):
				ret_ith_cell.append(i)
				break
	return ret_ith_cell

def get_range_cell_1D(l):
	ret_arr_range = []
	lowerBound = 0
	upperBound = l
	while(upperBound <= 1):
		ret_arr_range.append([lowerBound, upperBound])
		upperBound += l
		lowerBound += l
	return ret_arr_range

def nth_summation(nth_list, m):
	sum_nth = 0
	for i in range(0, len(nth_list)):
		sum_nth += iterative_m(nth_list[i], m)
	return sum_nth


def iterative_m(nth, m): # works
	ret_nth = 1
	for i in range(1, m+1):
		ret_nth *= nth - i + 1
	return ret_nth

'''
	create vertical line
'''
	
def get_sub_dims(dims, bucket_range):
	sub_dims = []
	sub_feat = []
	for feat in dims:
		for i in range(bucket_range[0], bucket_range[1]):
			sub_feat.append(feat[i])
		sub_dims.append(sub_feat)
		sub_feat = []
	return sub_dims




def append_data(dims, data):
	dims_tmp = list(map(list,dims))
	sub_dims = []
	sub_feat = []
	i = 0
	for fe in dims_tmp:
		for d in range(0, len(data[i])):
			fe.append(data[i][d])
		sub_dims.append(fe)
		i += 1
	return sub_dims

def pop_data(dims):
	sub_dims = []
	sub_feat = []
	for feat in dims:
		feat.pop()
		sub_dims.append(feat)
	return sub_dims


def merge_dims(dims1, dims2):
	ret_dims = list(map(list,dims1))
	for i in range(0, len(dims1)):
		for j in range(0, len(dims2[0])):
			ret_dims[i].append(dims2[i][j])
	return ret_dims