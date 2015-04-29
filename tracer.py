import sys
from sampler import *
INF = sys.maxint
epsilon = 1.0e-7

class Sphere(object):
    def __init__(self, center, radius, color):
        self.center = numpy.array(center)
        self.radius = numpy.array(radius)
        self.color = color

    def get_color(self):
        return self.color

    def hit(self, ray):
        temp = ray.origin - self.center
        a = numpy.dot(ray.direction, ray.direction)
        b = 2.0 * numpy.dot(temp, ray.direction)
        c = numpy.dot(temp, temp) - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        
        if (disc < 0.0):
            return None
        else:
            e = math.sqrt(disc)
            denom = 2.0 * a
            t = (-b - e) / denom
            # take one of the roots that actually is hit
            if (not t > epsilon):
                t = (-b + e) / denom
            if (t > epsilon):
                normal = (temp + t * ray.direction) / self.radius
                hit_point = ray.origin + t * ray.direction
                return ShadeRecord(normal=normal, hit_point=hit_point, tmin=t)

        return None    

class Plane(object):
    def __init__(self, origin, normal, color):
        self.origin = numpy.array(origin)
        self.normal = numpy.array(normal)
        self.color = color

    def get_color(self):
        return self.color
    def hit(self, ray):
        # ray is parallel to the plane
        if numpy.dot(ray.direction, self.normal) == 0:
            return None
        t = numpy.dot((self.origin - ray.origin) , self.normal) / numpy.dot(ray.direction, self.normal)
        if t > epsilon:
            hit_point = ray.origin + t * ray.direction
            return ShadeRecord(normal=self.normal, hit_point=hit_point, tmin=t)
        else:
            return None

class Ray(object):
    def __init__(self, origin, direction):
        self.origin = numpy.array(origin)
        self.direction = numpy.array(direction)

class ShadeRecord(object):
    def __init__(self, hit_point, normal, tmin):
        self.hit_point = hit_point
        self.normal = normal
        self.tmin = tmin

class Tracer(object):
    def __init__(self, world):
        self.world = world

    def trace_ray(self, ray):
        shader_rec = self.world.hit_bare_bones_objects(ray)
        if shader_rec:
            return shader_rec.color
        else:
            return (0.0,0.0,0.0)

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