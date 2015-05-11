import numpy
import sys
import random
from tracer import ShadeRecord

INF = sys.maxint
epsilon = 1.0e-7


class Sphere(object):
    def __init__(self, center, radius, material):
        self.center = numpy.array(center)
        self.radius = numpy.array(radius)
        self.material = material

    def get_material(self):
        return self.material

    def hit(self, ray):
        temp = ray.origin - self.center
        a = numpy.dot(ray.direction, ray.direction)
        b = 2.0 * numpy.dot(temp, ray.direction)
        c = numpy.dot(temp, temp) - self.radius * self.radius
        disc = b * b - 4.0 * a * c

        if (disc < 0.0):
            return None
        else:
            e = numpy.sqrt(disc)
            denom = 2.0 * a
            t = (-b - e) / denom
            # take one of the roots that actually is hit
            if (not t > epsilon):
                t = (-b + e) / denom
            if (t > epsilon):
                normal = (temp + t * ray.direction) / self.radius
                local_hit_point = ray.origin + t * ray.direction
                return ShadeRecord(normal=normal, local_hit_point=local_hit_point, tmin=t)

        return None

    def shadow_hit(self, ray):
        temp = ray.origin - self.center
        a = numpy.dot(ray.direction, ray.direction)
        b = 2.0 * numpy.dot(temp, ray.direction)
        c = numpy.dot(temp, temp) - self.radius * self.radius
        disc = b * b - 4.0 * a * c

        if (disc < 0.0):
            return False, 0
        else:
            e = numpy.sqrt(disc)
            denom = 2.0 * a
            t = (-b - e) / denom
            # take one of the roots that actually is hit
            if (not t > epsilon):
                t = (-b + e) / denom
            if (t > epsilon):
                return True, t
        return False, 0


class Plane(object):
    def __init__(self, origin, normal, material):
        self.origin = numpy.array(origin)
        self.normal = numpy.array(normal)
        self.material = material

    def get_material(self):
        return self.material

    def hit(self, ray):
        # ray is parallel to the plane
        if numpy.dot(ray.direction, self.normal) == 0:
            return None
        t = numpy.dot((self.origin - ray.origin), self.normal) / numpy.dot(ray.direction, self.normal)
        if t > epsilon:
            local_hit_point = ray.origin + t * ray.direction
            return ShadeRecord(normal=self.normal, local_hit_point=local_hit_point, tmin=t)
        else:
            return None

    def shadow_hit(self, ray):
        if numpy.dot(ray.direction, self.normal) == 0:
            return False, 0
        t = numpy.dot((self.origin - ray.origin), self.normal) / numpy.dot(ray.direction, self.normal)
        if t > epsilon:
            return True, t
        return False, 0


class Rectangle(object):
    def __init__(self, p0, a, b, normal, right, material, sampler):
        self.p0 = p0
        self.a = a
        self.b = b
        self.normal = normal
        self.right = right
        self.top = numpy.cross(normal, right)
        self.sampler = sampler
        self.inv_area = 1. / (a * b)
        self.material = material

    def sample(self):
        sample_point = random.choice(self.sampler.sample())
        return self.p0 + numpy.array((sample_point[0] * self.a, sample_point[1] * self.b, 0))

    def pdf(self, shader_rec):
        return self.inv_area

    def get_normal(self, sample_point):
        return self.normal

    def get_material(self):
        return self.material

    def hit(self, ray):
        # ray is parallel to the plane
        if numpy.dot(ray.direction, self.normal) == 0:
            return None
        t = numpy.dot((self.p0 - ray.origin), self.normal) / numpy.dot(ray.direction, self.normal)
        if t > epsilon:
            local_hit_point = ray.origin + t * ray.direction
            diff = local_hit_point - self.p0
            a = numpy.abs(diff.dot(self.right))
            b = numpy.abs(diff.dot(self.top))
            if a * 2 <= self.a and b * 2 <= self.b:
                return ShadeRecord(normal=self.normal, local_hit_point=local_hit_point, tmin=t)
        else:
            return None

    def shadow_hit(self, ray):
        if numpy.dot(ray.direction, self.normal) == 0:
            return False, 0
        t = numpy.dot((self.p0 - ray.origin), self.normal) / numpy.dot(ray.direction, self.normal)
        if t > epsilon:
            local_hit_point = ray.origin + t * ray.direction
            diff = local_hit_point - self.p0
            a = numpy.abs(diff.dot(self.right))
            b = numpy.abs(diff.dot(self.top))
            if a * 2 <= self.a and b * 2 <= self.b:
                return True, t
        return False, 0
