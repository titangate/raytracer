import numpy
import sys
import random
from tracer import ShadeRecord

INF = sys.maxint
epsilon = 1.0e-7


class BoundingBox(object):
    def __init__(self, x0, x1, y0, y1, z0, z1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def hit(self, ray):
        dx = 1. / ray.direction[0]
        dy = 1. / ray.direction[1]
        dz = 1. / ray.direction[2]

        t1 = (self.x0 - ray.origin[0]) * dx
        t2 = (self.x1 - ray.origin[0]) * dx
        t3 = (self.y0 - ray.origin[1]) * dy
        t4 = (self.y1 - ray.origin[1]) * dy
        t5 = (self.z0 - ray.origin[2]) * dz
        t6 = (self.z1 - ray.origin[2]) * dz

        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))

        if tmax < 0:
            return False

        if tmin > tmax:
            return False

        return True


class GeometryObject(object):
    def __init__(self):
        self.cast_shadow = True

    def get_material(self):
        return self.material


class AxisAlignedBox(GeometryObject):
    FACE_ARRAY = [numpy.array((-1,0,0)),
                  numpy.array((0,-1,0)),
                  numpy.array((0,0,-1)),
                  numpy.array((1,0,0)),
                  numpy.array((0,1,0)),
                  numpy.array((0,0,1)),]

    def __init__(self, x0, x1, y0, y1, z0, z1, material):
        super(AxisAlignedBox, self).__init__()
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.material = material

    def get_face_normal(self, face):
        return self.FACE_ARRAY[face]

    def hit(self, ray):
        dx = 1. / ray.direction[0]
        dy = 1. / ray.direction[1]
        dz = 1. / ray.direction[2]

        if dx >= 0:
            tx_min = (self.x0 - ray.origin[0]) * dx
            tx_max = (self.x1 - ray.origin[0]) * dx
        else:
            tx_min = (self.x1 - ray.origin[0]) * dx
            tx_max = (self.x0 - ray.origin[0]) * dx

        if dy >= 0:
            ty_min = (self.y0 - ray.origin[1]) * dy
            ty_max = (self.y1 - ray.origin[1]) * dy
        else:
            ty_min = (self.y1 - ray.origin[1]) * dy
            ty_max = (self.y0 - ray.origin[1]) * dy

        if dz >= 0:
            tz_min = (self.z0 - ray.origin[2]) * dz
            tz_max = (self.z1 - ray.origin[2]) * dz
        else:
            tz_min = (self.z1 - ray.origin[2]) * dz
            tz_max = (self.z0 - ray.origin[2]) * dz

        if tx_min > ty_min:
            t0 = tx_min
            if dx >= 0:
                face_in = 0
            else:
                face_in = 3
        else:
            t0 = ty_min
            if dy >= 0:
                face_in = 1
            else:
                face_in = 4

        if tz_min > t0:
            t0 = tz_min
            if dz >= 0:
                face_in = 2
            else:
                face_in = 5

        if tx_max < ty_max:
            t1 = tx_max
            if dx >= 0:
                face_out = 3
            else:
                face_out = 0
        else:
            t1 = ty_max
            if dy >= 0:
                face_out = 4
            else:
                face_out = 1

        if tz_max < t1:
            t1 = tz_max
            if dz >= 0:
                face_out = 5
            else:
                face_out = 2

        if t0 < t1 and t1 > epsilon:
            if t0 > epsilon:
                tmin = t0
                normal = self.get_face_normal(face_in)
            else:
                tmin = t1
                normal = self.get_face_normal(face_out)

            local_hit_point = ray.origin + tmin * ray.direction

            tmin = tmin
            shader_rec = ShadeRecord(normal=normal, local_hit_point=local_hit_point, tmin=tmin)
            return shader_rec
        else:
            return None

    def shadow_hit(self, ray):
        if not self.cast_shadow:
            return False, 0
        dx = 1. / ray.direction[0]
        dy = 1. / ray.direction[1]
        dz = 1. / ray.direction[2]

        t1 = (self.x0 - ray.origin[0]) * dx
        t2 = (self.x1 - ray.origin[0]) * dx
        t3 = (self.y0 - ray.origin[1]) * dy
        t4 = (self.y1 - ray.origin[1]) * dy
        t5 = (self.z0 - ray.origin[2]) * dz
        t6 = (self.z1 - ray.origin[2]) * dz

        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))

        if tmax < epsilon:
            return False, 0

        if tmin > tmax:
            return False, 0

        return True, tmin


class Sphere(GeometryObject):
    def __init__(self, center, radius, material, sampler=None):
        super(Sphere, self).__init__()
        self.center = numpy.array(center)
        self.radius = numpy.array(radius)
        self.material = material
        self.sampler = sampler

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
        if not self.cast_shadow:
            return False, 0
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

    def sample(self):
        pass


class ConcaveSphere(Sphere):
    def hit(self, ray):
        sr = super(ConcaveSphere, self).hit(ray)
        if sr:
            sr.normal = -sr.normal
        return sr


class Plane(GeometryObject):
    def __init__(self, origin, normal, material):
        super(Plane, self).__init__()
        self.origin = numpy.array(origin)
        self.normal = numpy.array(normal)
        self.material = material

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
        if not self.cast_shadow:
            return False, 0
        if numpy.dot(ray.direction, self.normal) == 0:
            return False, 0
        t = numpy.dot((self.origin - ray.origin), self.normal) / numpy.dot(ray.direction, self.normal)
        if t > epsilon:
            return True, t
        return False, 0


class Rectangle(GeometryObject):
    def __init__(self, p0, a, b, normal, right, material, sampler):
        super(Rectangle, self).__init__()
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
        if not self.cast_shadow:
            return False, 0
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


class Triangle(GeometryObject):
    def __init__(self, v0, v1, v2, material):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.normal = numpy.cross(v2 - v0, v1 - v0)
        self.normal /= numpy.linalg.norm(self.normal)
        self.material = material

    def shadow_hit(self, ray):
        d = self.v1 - self.v0
        e = self.v2 - self.v0
        f = self.v0 - ray.origin

        m = numpy.array((d,e,-ray.direction)).transpose()
        r = numpy.linalg.solve(m, -f)
        alpha, beta, tmin = r
        # import ipdb; ipdb.set_trace()
        if alpha > epsilon and beta > epsilon and tmin > epsilon and alpha + beta < 1:
            # import ipdb; ipdb.set_trace()
            return True, tmin

        return False, 0

    def hit(self, ray):
        d = self.v1 - self.v0
        e = self.v2 - self.v0
        f = self.v0 - ray.origin

        m = numpy.array((d,e,-ray.direction)).transpose()
        r = numpy.linalg.solve(m, -f)
        alpha, beta, t = r
        # import ipdb; ipdb.set_trace()
        if alpha > epsilon and beta > epsilon and t > epsilon and alpha + beta < 1:
            local_hit_point = ray.origin + t * ray.direction
            return ShadeRecord(normal=self.normal, local_hit_point=local_hit_point, tmin=t)

        return None
