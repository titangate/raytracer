from geometry import Mesh
import numpy as np
from material import Matte, Phong, Emissive
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

# 0. Color on and Ambient off
# 1. Color on and Ambient on
# 2. Highlight on
# 3. Reflection on and Ray trace on
# 4. Transparency: Glass on, Reflection: Ray trace on
# 5. Reflection: Fresnel on and Ray trace on
# 6. Transparency: Refraction on, Reflection: Fresnel off and Ray trace on
# 7. Transparency: Refraction on, Reflection: Fresnel on and Ray trace on
# 8. Reflection on and Ray trace off
# 9. Transparency: Glass on, Reflection: Ray trace off
# 10. Casts shadows onto invisible surfaces

class MaterialFactory(object):
    def __init__(self, f, mat_sampler):
        self.materials = {}

        def close_mat(name, params):
            Ka = params.get('Ka')
            Kd = params.get('Kd')
            Ks = params.get('Ks')
            Ke = params.get('Ke')
            Ns = params.get('Ns')
            Ni = params.get('Ni')
            illum = params.get('illum')

            color = Ka
            if 'Kd' in params:
                Kd = Kd[0] / color[0]
            if 'Ks' in params:
                Ks = Ks[0] / color[0]

            if int(illum) == 2:
                if 'Ke' in params and sum(Ke):
                    mat = Emissive(1., Ke)
                elif Ns and Ns != 0.:
                    mat = Matte(1., Kd, color, sampler=mat_sampler)
                elif Ns:
                    mat = Phong(Kd, color, Ni)

            print name, mat
            self.materials[name] = mat

        cur_mat = None
        params = {}
        for line in f:
            line = line.split()
            try:
                idx = line.index('#')
                line = line[:idx]
            except ValueError:
                pass
            if line:
                symbol = line[0]
                if symbol == 'newmtl':
                    if cur_mat is not None:
                        close_mat(cur_mat, params)
                        params = {}
                    cur_mat = line[1]
                else:
                    params[symbol] = np.array([float(a) for a in line[1:]])

        if params:
            close_mat(cur_mat, params)

    def get_material(self, name):
        return self.materials[name]


def read_mat(f, mat_sampler):
    return MaterialFactory(f, mat_sampler)


def read_mesh_complex(f, mat_sampler):
    vertices = []
    faces = []
    objects = {}
    current_key = None
    mtl = None
    mtl_lib = None
    folder = os.path.dirname(os.path.realpath(f))
    f = open(f)

    def close_mesh(vertices, faces, name, material):
        print name + ' using ' + material
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
                mtl_lib = read_mat(open(folder + '/' + line[1]), mat_sampler)

    if faces:
        objects[current_key] = close_mesh(vertices, faces, current_key, mtl)

    return objects
