import os

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def print_result(file, content):
	print(content)
	file.write(content)
	return file