import numpy
from tracer import Ray


class Material(object):
    def __init__(self):
        self.receives_shadow = True

    def get_Le(self):
        return numpy.array((0.,0.,0.))


class Lambertian(Material):
    def __init__(self, sampler, kd, cd):
        self.sampler = sampler
        self.kd = kd
        self.cd = cd

    def f(self, shader_rec, wo, wi):
        return self.kd * self.cd / numpy.pi

    def sample_f(self, shader_rec, wo, wi):
        w = shader_rec.normal
        v = numpy.cross(numpy.array([0.0034, 1, 0.0071]),w)
        v /= numpy.linalg.norm(v)
        u = numpy.cross(v,w)

        sp = self.sampler.sample_unit_hemisphere_surface()
        wi = sp[0] * u + sp[1] * v + sp[2] * w
        wi /= numpy.linalg.norm(wi)

        pdf = w.dot(wi) / numpy.pi

        return self.kd * self.cd / numpy.pi, wi, pdf

    def rho(self, shader_rec, wo):
        return self.kd * self.cd


class GlossySpecular(Material):
    def __init__(self, sampler, ks, cs, exp):
        self.sampler = sampler
        self.ks = ks
        self.cs = cs
        self.exp = exp

    def f(self, shader_rec, wo, wi):
        L = numpy.zeros(3)
        ndotwi = shader_rec.normal.dot(wi)
        r = -wi + 2.0 * shader_rec.normal * ndotwi  # incident vector

        rdotwo = r.dot(wo)
        if rdotwo > 0.0:
            L = self.ks * self.cs * rdotwo ** self.exp

        return L

    def rho(self, shader_rec, wo):
        return numpy.zeros(3)


class Matte(Material):
    def __init__(self, ka, kd, cd):
        self.ambient_brdf = Lambertian(None, ka, cd)
        self.diffuse_brdf = Lambertian(None, kd, cd)

    def set_ka(self, ka):
        self.ambient_brdf.kd = ka

    def set_kd(self, kd):
        self.diffuse_brdf.kd = kd

    def set_cd(self, c):
        self.ambient_brdf.cd = c
        self.diffuse_brdf.cd = c

    def shade(self, sr):
        wo = -sr.ray.direction
        L = self.ambient_brdf.rho(sr, wo) * sr.world.ambient_color.L(sr)
        for light in sr.world.lights:
            wi = light.get_direction(sr)
            ndotwi = sr.normal.dot(wi)

            if ndotwi > 0:
                in_shadow = False
                if light.cast_shadow:
                    shadow_ray = Ray(sr.hit_point, wi)
                    in_shadow = light.in_shadow(shadow_ray, sr)
                if not in_shadow and self.receives_shadow:
                    L += self.diffuse_brdf.f(sr, wo, wi) * light.L(sr) * ndotwi
        return L

    def area_light_shade(self, sr):
        wo = -sr.ray.direction
        L = self.ambient_brdf.rho(sr, wo) * sr.world.ambient_color.L(sr)
        for light in sr.world.lights:
            wi = light.get_direction(sr)
            ndotwi = sr.normal.dot(wi)

            if ndotwi > 0:
                in_shadow = False
                if light.cast_shadow:
                    shadow_ray = Ray(sr.hit_point, wi)
                    in_shadow = light.in_shadow(shadow_ray, sr)
                if not in_shadow and self.receives_shadow:
                    L += self.diffuse_brdf.f(sr, wo, wi) * light.L(sr) * ndotwi * light.G(sr) / light.pdf(sr)
        return L


class Phong(Material):
    def __init__(self, kd, cd, exp):
        self.ambient_brdf = Lambertian(None, kd, cd)
        self.diffuse_brdf = Lambertian(None, kd, cd)
        self.specular_brdf = GlossySpecular(None, kd, cd, exp)

    def set_ka(self, ka):
        self.ambient_brdf.kd = ka

    def set_kd(self, kd):
        self.diffuse_brdf.kd = kd

    def set_cd(self, c):
        self.ambient_brdf.cd = c
        self.diffuse_brdf.cd = c

    def shade(self, sr):
        wo = -sr.ray.direction
        L = self.ambient_brdf.rho(sr, wo) * sr.world.ambient_color.L(sr)
        for light in sr.world.lights:
            wi = light.get_direction(sr)
            ndotwi = sr.normal.dot(wi)

            if ndotwi > 0:
                in_shadow = False
                if light.cast_shadow:
                    shadow_ray = Ray(sr.hit_point, wi)
                    in_shadow = light.in_shadow(shadow_ray, sr)
                if not in_shadow and self.receives_shadow:
                    L += (self.diffuse_brdf.f(sr, wo, wi) + self.specular_brdf.f(sr, wo, wi)) * light.L(sr) * ndotwi
        return L

    def area_light_shade(self, sr):
        wo = -sr.ray.direction
        L = self.ambient_brdf.rho(sr, wo) * sr.world.ambient_color.L(sr)
        for light in sr.world.lights:
            wi = light.get_direction(sr)
            ndotwi = sr.normal.dot(wi)

            if ndotwi > 0:
                in_shadow = False
                if light.cast_shadow:
                    shadow_ray = Ray(sr.hit_point, wi)
                    in_shadow = light.in_shadow(shadow_ray, sr)
                if not in_shadow and self.receives_shadow:
                    L += ((self.diffuse_brdf.f(sr, wo, wi) + self.specular_brdf.f(sr, wo, wi))
                          * light.L(sr) * ndotwi * light.G(sr) / light.pdf(sr))
        return L


class Emissive(Material):
    def __init__(self, ls, color):
        self.ls = ls
        self.color = color

    def get_Le(self, shader_rec):
        return self.ls * self.color

    def shade(self, shader_rec):
        pass

    def area_light_shade(self, shader_rec):
        if -shader_rec.normal.dot(shader_rec.ray.direction) > 0:
            return self.ls * self.color
        else:
            return numpy.array((0.,0.,0.))
