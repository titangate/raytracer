from geometry import Mesh
import numpy as np


def read_mesh(f, face_limit=None):
    vertices = []
    indices = []
    for line in f:
        line = line.split()
        symbol = line[0]

        if symbol == 'v':
            vertices.append(np.array(map(float, line[1:])))
        if symbol == 'f':
            indices.append(map(lambda a:int(a) - 1, line[1:]))
    if face_limit:
        indices = indices[:face_limit]
    return Mesh(vertices, indices)
