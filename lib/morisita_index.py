import math
import sys
import lib.helper

''' MINDID original algorithm '''
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
		# sum_nth = 0
		cluster_m_sum = nth_summation(nth_list, m)
		# N_m_sum = nth_summation([len(dims[0])], m)


	###############
		MI =  (cluster_m_sum)
	###############

		# my_string = 'MI: '+str(MI)+'\n'
		# my_string = 'Q: '+str(Q)+'\n'
		# my_string = 'Sum(n): '+str(cluster_m_sum)+'\n'
		# my_string = 'Sum(N): '+str(N_m_sum)+'\n'
	
		ret_delta.append(delta)
		ret_MI.append(MI)


		delta = delta / 2


	return ret_MI, ret_delta

''' Divide the scaled data into a given benchmark
	and return cluster weights where m is a minimum
	cluster
'''
def Cluster_Sum(dims, m, benchmarks):

	ret_LC = []
	ret_delta = []
	delta = 1
	dims = [dims]
	feat_num = len(dims)

	for b_ith in range(1,benchmarks+1):
		Q = compute_Q_Multi_Dims(b_ith, feat_num)
		range_cell = get_range_cell_1D(delta)
		nth_list = get_multi_ith_cell(dims, range_cell, Q, m)
		sum_nth = 0
		cluster_m_sum = nth_summation(nth_list, m)
		N_m_sum = nth_summation([len(dims[0])], m)
		LC = cluster_m_sum
	
		ret_delta.append(delta)
		ret_LC.append(LC)

		delta = delta / 2

	return ret_LC, ret_delta

''' Returns list of overlapping cell
	return a list of each cell's weight. The
	return is a list type and does not need
	to be in order.
'''
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

	# Counting repeatitive hash (exact combination)
	# and increment its value. If combination
	# is 332, meaning 1st dimension is at cell 3,
	# 2nd dimension is at cell 3, 3rd dimension is at
	# cell 2
	for i in range(0, len(final_dim)):
		if final_dim[i] in cell_hash:
			cell_hash[final_dim[i]] += 1
		else:
			cell_hash[final_dim[i]] = 1

	# Filter out cells that do not contain more than
	# given m value
	for freq in cell_hash.values():
		if(freq >= m):
			n_ith_arr.append(freq)
	
	return (n_ith_arr)


'''
	Compute number of quadrats. This is computed by cells number of 1 dimension multiply by number of dimensions
'''
def compute_Q_Multi_Dims(b_ith, feat_num):
	return math.pow(2, ( (b_ith-1) * feat_num))

''' Get index of cells from the dimension
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

''' Get range of the cell from parameter l.
	For example, if l = 0.25, each cell
	will have a length of 0.25 equally and
	result in 4 cells in 0 to 1 value scaled data
'''
def get_range_cell_1D(l):
	ret_arr_range = []
	lowerBound = 0
	upperBound = l
	while(upperBound <= 1):
		ret_arr_range.append([lowerBound, upperBound])
		upperBound += l
		lowerBound += l
	return ret_arr_range

''' Sum up of summation n(n-1)(n-m)...
'''
def nth_summation(nth_list, m):
	sum_nth = 0
	for i in range(0, len(nth_list)):
		sum_nth += iterative_m(nth_list[i], m)
	return sum_nth

''' Iteratively removing candidate clusters
	that do not have above m cluster. The
	algorithm is given on the paper
'''
def iterative_m(nth, m):
	ret_nth = 1
	for i in range(1, m+1):
		ret_nth *= nth - i + 1
	return ret_nth

