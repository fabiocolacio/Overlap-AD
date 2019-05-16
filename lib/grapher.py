import lib.dir_manager
import lib.scaler
import csv
import datetime
import time
from datetime import timezone
import numpy as np
import math
import matplotli	b.pyplot as plt
fig = plt.figure()

matplot_color = ['b','g','r','c','m','y','k','w']

def create_skeleton(output_name, dims_no):
	plt.axis([0,1,0,dims_no+1])
	plt.locator_params('y', nbins=5)
	plt.locator_params('x', nbins=10)

	for i in range(0, dims_no):
		plt.plot([0,1],[i+1,i+1], linewidth=1)
	# INPUT
	fig.savefig('./'+output_name, dpi=500)
	pass

def benchmark_division(output_name, dims_no, arr_div):
	for div in arr_div:
		plt.plot([div,div], [0,dims_no+1], 'k', linewidth=1)
	fig.savefig('./'+output_name, dpi=500)
	pass

def plot_data(output_name, dims, bucket_range):
	for feat_no in range(0,len(dims)):
		data_num = 0;
		for i in range(bucket_range[0], bucket_range[1]):
			plt.plot([dims[feat_no][i]], [feat_no+1],'x')
			# print(matplot_color[data_num])
			data_num += 1
	fig.savefig('./'+output_name, dpi=500)
	plt.cla()
	plt.clf()
	pass

def plot_MI(output_name, MI_array):
	plt.cla()
	plt.clf()
	for i in range(0, len(MI_array[0])):
		plt.plot(MI_array[0][i], MI_array[1][i], 'bo')

		
	plt.plot([0,1,2,3,4,5], [1,1,1,1,1,1], 'g')
	plt.axis([0,1.5,0, 5000])	
	plt.plot(MI_array[0], MI_array[1], 'b')
	mkdir('graphs_MI')
	fig.savefig('graphs_MI/MI_graph', dpi=500)
	plt.cla()
	plt.clf()