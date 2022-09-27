import os
import random

import h5py
import numpy as np
from plyfile import PlyData, PlyElement
from multiprocessing import Pool

basedir = os.path.dirname(__file__)
dataset = os.path.join(basedir, "Dataset")
shapeNetdir = os.path.join(dataset, "ShapeNet")
filenames = [line.rstrip() for line in open(os.path.join(dataset, "autoSliceFiles.txt"), 'r')]
fileQuantity = 1
horizontalGrids = 1
verticalGrids = 384
h5verticesSize = 2048
verticesZoomRatio = 16

class Vertex:
    def __init__(self, x, y, z, label):
        self.x = x
        self.y = y
        self.z = z
        self.label = label

    def __str__(self):
        return f"{self.x}, {self.y}, {self.z}: {self.label}"


class Grid:
    def __init__(self, lowerX, lowerY, upperX, upperY):
        self.lowerX = float(lowerX)
        self.lowerY = float(lowerY)
        self.upperX = float(upperX)
        self.upperY = float(upperY)
        self.vertices = []

    def isInRange(self, vertex):
        return self.lowerX <= vertex.x < self.upperX and self.lowerY <= vertex.y < self.upperY

    def pushVertex(self, vertex):
        if self.isInRange(vertex):
            self.vertices.append(vertex)

    def __str__(self):
        return f"x:{self.lowerX} - {self.upperX}, y:{self.lowerY} - {self.upperY}, v:{len(self.vertices)}"


def isMagenta(vertices, index):
    if (vertices['red'][index] >= 201 and vertices['green'][index] <= 71 and vertices['blue'][index] >= 245):
        return 1
    return 0


def randomArrange(vertices):
    random.shuffle(vertices)
    vs = []
    prevIndex = 0
    if len(vertices) > h5verticesSize:
        for i in range(h5verticesSize, len(vertices) + 1, h5verticesSize):
            vs.append(vertices[prevIndex: i])
            prevIndex = i

    if len(vertices) % h5verticesSize != 0:
        restVertices = vertices[-len(vertices) % h5verticesSize:]
        random.shuffle(vertices)
        vs.append(restVertices + vertices[:h5verticesSize - len(restVertices)])

    vertices.clear()
    for slices in vs:
        vertices.append(slices)


def processData(filename):
    f = h5py.File(os.path.join(dataset, f"{filename[:-4]}_{h5verticesSize}x{verticesZoomRatio}.h5"), 'w')
    plydata = PlyData.read(os.path.join(dataset, filename))
    totalv = plydata['vertex'].count

    maxX = -2147483648
    maxY = -2147483648
    minX = 2147483647
    minY = 2147483647
    vertices = []
    grids = []
    for j in range(0, totalv):
        x = plydata['vertex']['x'][j]
        y = plydata['vertex']['y'][j]
        z = plydata['vertex']['z'][j]
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y
        newV = Vertex(x, y, z, isMagenta(plydata['vertex'], j))
        vertices.append(newV)

    widthInterval = (maxX - minX) / horizontalGrids
    heightInterval = (maxY - minY) / verticalGrids
    prevX = minX
    prevY = minY
    currentX = minX + widthInterval
    currentY = minY + heightInterval
    for x in range(horizontalGrids):
        for y in range(verticalGrids):
            grids.append(Grid(prevX, prevY, currentX, currentY))
            prevY = currentY
            currentY += heightInterval
        prevX = currentX
        currentX += widthInterval
        prevY = minY
        currentY = minY + heightInterval

    for v in vertices:
        for g in range(len(grids)):
            grids[g].pushVertex(v)

    vertices = []
    for g in grids:
        for v in g.vertices:
            vertices.append(v)

    slices = totalv // h5verticesSize + min(totalv % h5verticesSize, 1)

    a_data = np.zeros((slices, h5verticesSize, 3))
    a_pid = np.zeros((slices, h5verticesSize), dtype=np.uint8)
    a_label = np.zeros((slices, 1), dtype=np.uint8)

    sliceIndex = 0
    startIndex = 0
    endIndex = startIndex + h5verticesSize * verticesZoomRatio
    while startIndex < totalv:
        if endIndex <= totalv:
            continueGrids = vertices[startIndex: endIndex]
        else:
            continueGrids = vertices[startIndex:]
        if len(continueGrids) >= h5verticesSize:
            randomArrange(continueGrids)
        else:
            repeatVertices = vertices[-(h5verticesSize * verticesZoomRatio): startIndex]
            random.shuffle(repeatVertices)
            continueGrids = [continueGrids + repeatVertices]
        for aSlice in continueGrids:
            for j in range(0, h5verticesSize):
                a_data[sliceIndex, j] = [aSlice[j].x, aSlice[j].y, aSlice[j].z]
                a_pid[sliceIndex, j] = aSlice[j].label
            sliceIndex += 1
        startIndex = endIndex
        endIndex = startIndex + h5verticesSize * verticesZoomRatio

    f.create_dataset("data", data=a_data)
    f.create_dataset("label", data=a_label)
    f.create_dataset("pid", data=a_pid)
    f.close()
    print(f"Finished{filename}")

if __name__ == '__main__':
    with Pool(len(filenames)) as p:
        p.map(processData, filenames)