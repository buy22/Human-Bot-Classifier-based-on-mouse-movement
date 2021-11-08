import numpy as np


def read_train_data():
	train_path = "data/dsjtzs_txfz_training.txt"
	id, trajectory, destination, label = [], [], [], []
	with open(train_path,"r") as f:
		for line in f:
			line = line.split()
			id.append(line[0])
			trajectory.append(line[1])
			destination.append(line[2])
			label.append(line[3])

	x, y, t = [], [], []
	for i in range(len(trajectory)):
		xyt = trajectory[i].split(";")[:-1]
		x.append([])
		y.append([])
		t.append([])
		for j in range(len(xyt)):
			tmp = xyt[j].split(",")
			x[i].append(float(tmp[0]))
			y[i].append(float(tmp[1]))
			t[i].append(float(tmp[2]))

	x_d, y_d = [], []
	for i in range(len(destination)):
		xy = destination[i].split(",")
		x_d.append(float(xy[0]))
		y_d.append(float(xy[1]))

	id = np.array(id).astype(int)
	label = np.array(label).astype(int)
	return id, x, y, t, x_d, y_d, label


def read_test_data():
	test_path = "data/dsjtzs_txfz_test1.txt"
	id, trajectory, destination = [], [], []
	with open(test_path,"r") as f:
		for line in f:
			line = line.split()
			id.append(line[0])
			trajectory.append(line[1])
			destination.append(line[2])
	x, y, t = [], [], []
	for i in range(len(trajectory)):
		xyt = trajectory[i].split(";")[:-1]
		x.append([])
		y.append([])
		t.append([])
		for j in range(len(xyt)):
			tmp = xyt[j].split(",")
			x[i].append(float(tmp[0]))
			y[i].append(float(tmp[1]))
			t[i].append(float(tmp[2]))

	x_d, y_d = [], []
	for i in range(len(destination)):
		xy = destination[i].split(",")
		x_d.append(float(xy[0]))
		y_d.append(float(xy[1]))

	id = np.array(id).astype(int)
	return id, x, y, t, x_d, y_d
