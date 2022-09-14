import os

import h5py
import numpy as np
from plyfile import PlyData, PlyElement

basedir = os.path.dirname(__file__)
dataset = os.path.join(basedir, "Dataset")
shapeNetdir = os.path.join(dataset, "ShapeNet")
filenames = [line.rstrip() for line in open(os.path.join(dataset, "plyFiles.txt"), 'r')]

#f = h5py.File("./hdf5_data/data_training.h5", 'w')
f = h5py.File(os.path.join(dataset, "data_testing.h5"), 'w')
vsize = 2048
totalv = 749637
batch = totalv // vsize
a_data = np.zeros((batch, vsize, 3))
a_pid = np.zeros((batch, vsize), dtype = np.uint8)
a_label = np.zeros((batch, 1), dtype = np.uint8)

def isMagenta(vertex, index):
	if(vertex['red'][index] >= 201 and vertex['green'][index] <= 71 and vertex['blue'][index] >= 245):
		return 1
	return 0
i = 0
lower = 0
upper = lower + vsize
plydata = PlyData.read(os.path.join(dataset, filenames[i] + ".ply"))

while(i < batch):
	#piddata = [line.rstrip() for line in open((os.path.join(dataset, filenames[i] + ".seg")), 'r')]
	for j in range(lower, upper):
		a_data[i, j - lower] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]
		#a_pid[i,j] = piddata[j]
		a_pid[i, j - lower] = isMagenta(plydata['vertex'], j)
		#a_label[i, j - lower] = 0

	i += 1
	lower += vsize
	upper += vsize

data = f.create_dataset("data", data = a_data)
label = f.create_dataset("label", data = a_label)
pid = f.create_dataset("pid", data = a_pid)