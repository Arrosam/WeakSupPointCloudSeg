import os

import h5py

def colorById(pid):
    if(pid == 0):
        return '0 0 0'
    if(pid == 1):
        return '255 71 255'

basedir = os.path.dirname(__file__)
dataset = os.path.join(basedir, "Dataset")
outputDir = os.path.join(dataset, "hdf5ToPly")
filename = os.path.join(dataset, "autoSlice2048.h5")
with h5py.File(filename, 'r') as h5f:
    ply_ds = h5f['data']
    ply_ps = h5f['pid']
    print(ply_ds.shape, ply_ds.dtype)
    for count in range(len(ply_ds)):
        v_cnt = len(ply_ds[count])
        row = ply_ds[count]
        pid_row = ply_ps[count]
        with open(os.path.join(outputDir, f'pcds_{str(count)}.ply'),'w') as ply_f:
            ply_f.write('ply\n')
            ply_f.write('format ascii 1.0\n')
            ply_f.write(f'comment row {count} exported from:{filename}\n')
            ply_f.write(f'element vertex {v_cnt}\n') # variable # of vertices
            ply_f.write('property float x\n')
            ply_f.write('property float y\n')
            ply_f.write('property float z\n')
            ply_f.write('property uchar red\n')
            ply_f.write('property uchar green\n')
            ply_f.write('property uchar blue\n')
            ply_f.write('end_header\n')
            for vertex_index in range(v_cnt):
                ply_f.write(f'{row[vertex_index][0]} '
                            f'{row[vertex_index][1]} '
                            f'{row[vertex_index][2]} '
                            f'{colorById(pid_row[vertex_index])}\n')