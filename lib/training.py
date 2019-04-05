import sys
import random
import lib.helper
import lib.morisita_index as mi

'''
	Randomize the given data
	Param: dims = dimensions, percentage = percentage of data to be randomized
	Return: list of random data in given percentage on each dimension
'''
def randomize_data(dims, data_num):
	dims_instant = list(map(list,dims))
	randomized_dims = []
	tmp_dims = []
	randomized_index = random.sample(range(len(dims[0])), data_num)


	for feat in dims_instant:
		for d in range(0, len(randomized_index)):
			tmp_dims.append(feat[randomized_index[d]])
		randomized_dims.append(tmp_dims)
		tmp_dims = []

	return randomized_dims

'''
	Perform a tenfold cross validation. 
	Param: dims = dimensions, percentage = percentage of data to be randomized
	Return: A list of data selected in randomized data
'''
def tenfold_cross_validation_random(dims, percentage):
	tenfold = []
	for i in range(0,10):
		tenfold.append(randomize_data(dims, int((percentage/100)*len(dims[0])) ))
	return tenfold


def tenfold_cross_validation(dims):
	tenfold = []
	high = int(len(dims[0])/10)
	for i in range(0,10):
		# training_range = [ [0, low+(high*i)], [low+(high*i)+int(0.1*len(dims[0])), len(dims[0])]]
		training_range = [ [0, (high*i)]    , [(high*i)+high, len(dims[0])]]
		
		dims1 = mi.get_sub_dims(dims, training_range[0])
		dims2 = mi.get_sub_dims(dims, training_range[1])
		tenfold.append(mi.merge_dims(dims1,dims2))
		print('dims1: ', dims1)
		print('dims2: ', dims2)
		print('Training Dimension: ',mi.merge_dims(dims1,dims2))
		print('Range Taken: ' ,training_range[0], ' ', training_range[1])
		print('Length: ', len(mi.merge_dims(dims1,dims2)[0]))
		print('\n\n')
	return tenfold


def hundredfold_cross_validation(dims):
	hundredfold = []
	high = int(len(dims[0])/5)
	for i in range(0,5):
		# training_range = [ [0, low+(high*i)], [low+(high*i)+int(0.1*len(dims[0])), len(dims[0])]]
		training_range = [ [0, (high*i)]    , [(high*i)+high, len(dims[0])]]
		
		dims1 = mi.get_sub_dims(dims, training_range[0])
		dims2 = mi.get_sub_dims(dims, training_range[1])
		hundredfold.append(mi.merge_dims(dims1,dims2))
		print('dims1: ', dims1)
		print('dims2: ', dims2)
		print('Training Dimension: ',mi.merge_dims(dims1,dims2))
		print('Range Taken: ' ,training_range[0], ' ', training_range[1])
		print('Length: ', len(mi.merge_dims(dims1,dims2)[0]))
		print('\n\n')
	return hundredfold