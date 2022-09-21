import os

import h5py
import numpy as np
from plyfile import PlyData, PlyElement

basedir = os.path.dirname(__file__)
dataset = os.path.join(basedir, "Dataset")
filenames = [line.rstrip() for line in open(os.path.join(dataset, "plyFilesEvaluating.txt"), 'r')]

#f = h5py.File("./hdf5_data/data_training.h5", 'w')
f = h5py.File(os.path.join(dataset, "multiSegEva.h5"), 'w')
vsize = 2048
batch = len(filenames)
a_data = np.zeros((batch, vsize, 3))
a_pid = np.zeros((batch, vsize), dtype = np.uint8)
a_label = np.zeros((batch, 1), dtype = np.uint8)

def isMagenta(vertex, index):
	if(vertex['red'][index] >= 201 and vertex['green'][index] <= 71 and vertex['blue'][index] >= 245):
		return 1
	return 0

for i in range(0, len(filenames)):
	plydata = PlyData.read(os.path.join(dataset, filenames[i] + ".ply"))
	for j in range(0, 2048):
		a_data[i, j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]
		a_pid[i, j] = isMagenta(plydata['vertex'], j)

data = f.create_dataset("data", data = a_data)
label = f.create_dataset("label", data = a_label)
pid = f.create_dataset("pid", data = a_pid)