import numpy
from tracer import Ray


class AmbientLight(object):
    def __init__(self, color, ls):
        self.color = color
        self.ls = ls
        self.cast_shadow = False

    def get_direction(self, shader_rec):
        return numpy.zeros(3)

    def L(self, shader_rec):
        return self.ls * self.color


class PointLight(object):
    def __init__(self, color, ls, location, attenuation=0, radius=300, cast_shadow=False):
        self.color = color
        self.ls = ls
        self.location = location
        self.attenuation = attenuation
        self.radius = radius
        self.cast_shadow = cast_shadow

    def get_direction(self, shader_rec):
        v = self.location - shader_rec.hit_point
        return v / numpy.linalg.norm(v)

    def in_shadow(self, ray, shader_rec):
        d = numpy.linalg.norm(ray.origin - self.location)
        for obj in shader_rec.world.objects:
            is_hit, t = obj.shadow_hit(ray)
            if is_hit and t < d:
                return True
        return False

    def L(self, shader_rec):
        color = self.ls * self.color
        if self.attenuation != 0:
            color *= (numpy.linalg.norm(shader_rec.hit_point - self.location) / self.radius) ** -self.attenuation

        return color


class DirectionLight(object):
    def __init__(self, color, ls, direction, cast_shadow=False):
        self.color = color
        self.ls = ls
        self.direction = direction
        self.cast_shadow = cast_shadow

    def get_direction(self, shader_rec):
        return self.direction

    def L(self, shader_rec):
        return self.ls * self.color

    def in_shadow(self, ray, shader_rec):
        for obj in shader_rec.world.objects:
            is_hit, t = obj.shadow_hit(ray)
            if is_hit:
                return True
        return False


class AmbientOccluder(object):
    def __init__(self, color, ls, sampler):
        self.color = color
        self.ls = ls
        self.sampler = sampler

    def get_direction(self, shader_rec):
        sample = self.sampler.sample_unit_hemisphere_surface()
        return self.u * sample[0] + self.v * sample[1] + self.w * sample[2]

    def in_shadow(self, ray, shader_rec):
        for obj in shader_rec.world.objects:
            is_hit, t = obj.shadow_hit(ray)
            if is_hit:
                return True
        return False

    def L(self, shader_rec):
        w = shader_rec.normal
        v = numpy.cross(w, numpy.array((0.0072, 1.0, 0.0034)))
        v /= numpy.linalg.norm(v)
        u = numpy.cross(v, w)

        self.u = u
        self.v = v
        self.w = w

        shadow_ray = Ray(shader_rec.hit_point, self.get_direction(shader_rec))

        if self.in_shadow(shadow_ray, shader_rec):
            return numpy.array((0.1,0.1,0.1)) * self.ls * self.color
        else:
            return self.ls * self.color


class AreaLight(object):
    def __init__(self, color, ls, material, shape, cast_shadow=True):
        self.color = color
        self.ls = ls
        self.material = material
        self.shape = shape
        self.cast_shadow = cast_shadow

    def L(self, shader_rec):
        if self.light_normal.dot(self.wi) < 0.:
            return self.material.get_Le(shader_rec)
        else:
            return numpy.array((0.0,0.0,0.0))

    def get_direction(self, shader_rec):
        self.sample_point = self.shape.sample()
        self.light_normal = self.shape.get_normal(self.sample_point)
        self.wi = self.sample_point - shader_rec.hit_point
        self.wi /= numpy.linalg.norm(self.wi)

        return self.wi

    def in_shadow(self, ray, shader_rec):
        d = (self.sample_point - ray.origin).dot(ray.direction)
        for obj in shader_rec.world.objects:
            is_hit, t = obj.shadow_hit(ray)
            if is_hit and t < d:
                return True
        return False

    def G(self, shader_rec):
        ndotd = -self.light_normal.dot(self.wi)
        d2 = self.sample_point - shader_rec.hit_point
        d2 = sum(d2 ** 2)

        return ndotd / d2

    def pdf(self, shader_rec):
        return self.shape.pdf(shader_rec)


class EnvironmentLight(object):
    def __init__(self, material, sampler, cast_shadow=True):
        self.material = material
        self.sampler = sampler
        self.cast_shadow = cast_shadow

    def G(self, shader_rec):
        return 1.

    def get_direction(self, shader_rec):
        w = shader_rec.normal
        v = numpy.cross(w, numpy.array((0.0072, 1.0, 0.0034)))
        v /= numpy.linalg.norm(v)
        u = numpy.cross(v, w)

        self.u = u
        self.v = v
        self.w = w

        sample = self.sampler.sample_unit_hemisphere_surface()
        self.wi = self.u * sample[0] + self.v * sample[1] + self.w * sample[2]

        return self.wi

    def L(self, shader_rec):
        return self.material.get_Le(shader_rec)

    def in_shadow(self, ray, shader_rec):
        for obj in shader_rec.world.objects:
            is_hit, t = obj.shadow_hit(ray)
            if is_hit:
                return True
        return False

    def pdf(self, shader_rec):
        return 1.
