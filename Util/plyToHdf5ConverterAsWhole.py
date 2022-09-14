import os

import h5py
import numpy as np
from plyfile import PlyData, PlyElement

basedir = os.path.dirname(__file__)
dataset = os.path.join(basedir, "Dataset")
shapeNetdir = os.path.join(dataset, "ShapeNet")
filenames = [line.rstrip() for line in open(os.path.join(dataset, "plyFiles.txt"), 'r')]
fileQuantity = len(filenames)

f = h5py.File(os.path.join(dataset, "data_testing_whole.h5"), 'w')
totalv = 749637
a_data = np.zeros((fileQuantity, totalv, 3))
a_pid = np.zeros((fileQuantity, totalv), dtype = np.uint8)
a_label = np.zeros((fileQuantity, 1), dtype = np.uint8)

def isMagenta(vertex, index):
	if(vertex['red'][index] >= 201 and vertex['green'][index] <= 71 and vertex['blue'][index] >= 245):
		return 1
	return 0
i = 0
plydata = PlyData.read(os.path.join(dataset, filenames[i] + ".ply"))
for j in range(0, totalv):
    a_data[0, j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]
    a_pid[0, j] = isMagenta(plydata['vertex'], j)

data = f.create_dataset("data", data = a_data)
label = f.create_dataset("label", data = a_label)
pid = f.create_dataset("pid", data = a_pid)