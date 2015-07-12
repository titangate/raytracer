import sys
from sampler import *
INF = sys.maxint
epsilon = 1.0e-7


class Ray(object):
    def __init__(self, origin, direction):
        self.origin = numpy.array(origin)
        self.direction = numpy.array(direction)


class ShadeRecord(object):
    def __init__(self, local_hit_point, normal, tmin, material=None):
        self.local_hit_point = local_hit_point
        self.normal = normal
        self.tmin = tmin
        self.material = material


class Tracer(object):
    def __init__(self, world, depth_limit=5):
        self.world = world
        self.depth_limit = depth_limit

    def trace_ray(self, ray, depth=0):
        if depth > self.depth_limit:
            return self.world.background_color
        shader_rec = self.world.hit_objects(ray)
        if shader_rec:
            shader_rec.ray = ray
            shader_rec.depth = depth
            return shader_rec.material.shade(shader_rec)
        else:
            return self.world.background_color


class WhittedTracer(object):
    def __init__(self, world, depth_limit=5):
        self.world = world
        self.depth_limit = depth_limit

    def trace_ray(self, ray, depth=0, tminlist=None):
        if depth > self.depth_limit:
            if tminlist:
                tminlist[0] = 0
            return self.world.background_color
        shader_rec = self.world.hit_objects(ray)
        if shader_rec:
            shader_rec.ray = ray
            shader_rec.depth = depth
            if tminlist:
                tminlist[0] = shader_rec.tmin
            return shader_rec.material.shade(shader_rec)
        else:
            if tminlist:
                tminlist[0] = 0
            return self.world.background_color


class AreaLightTracer(object):
    def __init__(self, world):
        self.world = world

    def trace_ray(self, ray, depth=0):
        shader_rec = self.world.hit_objects(ray)
        if shader_rec:
            shader_rec.ray = ray
            return shader_rec.material.area_light_shade(shader_rec)
        else:
            return (0.0, 0.0, 0.0)


class PathTracer(object):
    def __init__(self, world, depth_limit=5):
        self.world = world
        self.depth_limit = depth_limit

    def trace_ray(self, ray, depth=0):
        if depth > self.depth_limit:
            return (0.0,0.0,0.0)
        shader_rec = self.world.hit_objects(ray)
        if shader_rec:
            shader_rec.ray = ray
            shader_rec.depth = depth
            return shader_rec.material.path_shade(shader_rec)
        else:
            return (0.0, 0.0, 0.0)


class ViewPlane(object):
    def __init__(self, resolution, pixel_size, sampler=RegularSampler):
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.sampler = sampler

    def iter_row(self, row):
        for column in xrange(self.resolution[0]):
            yield (column,row)

    def __iter__(self):
        for row in xrange(self.resolution[1]):
            yield self.iter_row(row)
