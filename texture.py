from PIL import Image
import numpy as np
import math


class Texture(object):
    def __init__(self):
        pass


class ImageTexture(Texture):
    def __init__(self, image, color, mapping, hres=None, vres=None):
        self.image = image
        self.color = color
        self.mapping = mapping
        self.hres = hres
        self.vres = vres
        self.instantiated = False

    def instantiate_for_proc(self):
        image = self.image
        hres = self.hres
        vres = self.vres
        if type(image) == str:
            image = Image.open(image)
        if hres is None:
            hres, vres = image.size
        self.hres = hres
        self.vres = vres
        self.instantiated = True
        self.image = image

    def get_color(self, sr):
        if not self.instantiated:
            self.instantiate_for_proc()
        if self.mapping:
            column, row = self.mapping.get_texel_coordinates(
                local_hit_point=sr.local_hit_point,
                hres=self.hres,
                vres=self.vres)
        else:
            column = int((self.hres - 1) * sr.u)
            row = int((self.vres - 1) * sr.v)

        color = np.array(self.image.getpixel((column, row)), float) / 256.
        return color


class Mapping(object):
    pass


class SphericalMap(Mapping):
    def get_texel_coordinates(self, local_hit_point, hres, vres):
        # compute theta and phi
        theta = math.acos(local_hit_point[1])
        phi = math.atan2(local_hit_point[0], local_hit_point[2])
        if phi < 0.:
            phi += math.pi * 2

        # map theta and phi to uv
        u = phi / math.pi / 2
        v = 1 - theta / math.pi

        # map uv to texel coords
        column = int((hres - 1) * u)
        row = int((vres - 1) * v)
        return column, row
