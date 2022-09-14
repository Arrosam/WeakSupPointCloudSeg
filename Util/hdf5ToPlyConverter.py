import os

import h5py

basedir = os.path.dirname(__file__)
dataset = os.path.join(basedir, "Dataset")
shapeNetdir = os.path.join(dataset, "ShapeNet")
filename = os.path.join(dataset, "ply_data_val0.h5")
with h5py.File(filename, 'r') as h5f:
    ply_ds = h5f['data']
    print(ply_ds.shape, ply_ds.dtype)
    for cnt, row in enumerate(ply_ds):
        v_cnt = row.shape[0]
        with open(os.path.join(dataset, f'pcds_{str(cnt)}.ply'),'w') as ply_f:
            ply_f.write('ply\n')
            ply_f.write('format ascii 1.0\n')
            ply_f.write(f'comment row {cnt} exported from:{filename}\n')
            ply_f.write(f'element vertex {v_cnt}\n') # variable # of vertices
            ply_f.write('property float x\n')
            ply_f.write('property float y\n')
            ply_f.write('property float z\n')
            ply_f.write('end_header\n')
            for vertex in row:
                ply_f.write(f'{vertex[0]:#6.3g} {vertex[1]:#6.3g} {vertex[2]:#6.3g}\n')