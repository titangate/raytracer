from geometry import Mesh
import numpy as np
from material import Matte
import os


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


class MaterialFactory(object):
    def get_material(self, name):
        return Matte(1, 1, np.array((1., 1., 1.)))


def read_mat(f):
    return MaterialFactory()


def read_mesh_complex(f):
    vertices = []
    faces = []
    objects = {}
    current_key = None
    mtl = None
    mtl_lib = None
    folder = os.path.dirname(os.path.realpath(f))
    f = open(f)

    def close_mesh(vertices, faces, name, material):
        mesh = Mesh(vertices, faces, mtl_lib.get_material(material))
        print faces, name
        return mesh

    for line in f:
        line = line.split()
        if line:
            symbol = line[0]
            if symbol == 'v':
                if current_key is not None:
                    objects[current_key] = close_mesh(vertices, faces, current_key, mtl)
                    faces = []
                    current_key = None
                vertices.append(np.array(map(float, line[1:])))
            elif symbol == 'f':
                faces.append(
                    map(
                        lambda a:int(a) - 1 if int(a) > 0 else len(vertices) + int(a),
                        line[1:]
                    )
                )
            elif symbol == 'g':
                current_key = line[1]
            elif symbol == 'usemtl':
                mtl = line[1]
            elif symbol == 'mtllib':
                mtl_lib = read_mat(open(folder + '/' + line[1]))

    if faces:
        close_mesh(vertices, faces, current_key, mtl)

    return objects