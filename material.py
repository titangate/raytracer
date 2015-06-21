import numpy
from tracer import Ray


class Material(object):
    def __init__(self):
        self.receives_shadow = True

    def get_Le(self):
        return numpy.array((0.,0.,0.))


class BDRF(object):
    pass


class Lambertian(BDRF):
    def __init__(self, sampler, kd, cd):
        super(Lambertian, self).__init__()
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


class PerfectSpecular(BDRF):
    def __init__(self, sampler, kd, cd):
        super(PerfectSpecular, self).__init__()
        self.sampler = sampler
        self.kd = kd
        self.cd = cd

    def f(self, shader_rec, wo, wi):
        return self.kd * self.cd / numpy.pi

    def sample_f(self, shader_rec, wo):
        ndotwo = shader_rec.normal.dot(wo)
        wi = -wo + 2.0 * shader_rec.normal * ndotwo

        return self.kd * self.cd / (shader_rec.normal.dot(wi)), wi

    def rho(self, shader_rec, wo):
        return self.kd * self.cd


class GlossySpecular(BDRF):
    def __init__(self, sampler, ks, cs, exp):
        super(GlossySpecular, self).__init__()
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

    def sample_f(self, shader_rec, wo):
        ndotwo = shader_rec.normal.dot(wo)
        w = -wo + 2.0 * shader_rec.normal * ndotwo

        u = numpy.cross(numpy.array((0.00424, 1., 0.00767)), w)
        u /= numpy.linalg.norm(u)
        v = numpy.cross(u, w)

        sp = self.sampler.sample_unit_hemisphere_surface()
        wi = sp[0] * u + sp[1] * v + sp[2] * w

        if shader_rec.normal.dot(wi) < 0.0:
            wi = -sp[0] * u - sp[1] * v + sp[2] * w

        phong_lobe = (w.dot(wi)) ** self.exp
        pdf = phong_lobe * (shader_rec.normal.dot(wi))
        if pdf == 0:
            return self.cs * 0, wi, 1

        return self.ks * self.cs * phong_lobe, wi, pdf


class Matte(Material):
    def __init__(self, ka, kd, cd):
        super(Matte, self).__init__()
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
                if not in_shadow:
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
    def __init__(self, kd, cd, exp, ka=None):
        super(Phong, self).__init__()
        if ka is None:
            ka = kd
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


class Reflective(Phong):
    def __init__(self, kd, cd, exp, ka=None, kf=None):
        super(Reflective, self).__init__(kd, cd, exp, ka)
        if kf is None:
            kf = kd
        self.reflective_bdrf = PerfectSpecular(None, kf, cd)

    def shade(self, sr):
        L = super(Reflective, self).shade(sr)

        wo = -sr.ray.direction
        fr, wi = self.reflective_bdrf.sample_f(sr, wo)

        reflected_ray = Ray(sr.hit_point, wi)

        L += fr * sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1) * sr.normal.dot(wi)
        return L


class GlossyReflective(Phong):
    def __init__(self, kd, cd, exp, sampler, ka=None, kf=None):
        super(GlossyReflective, self).__init__(kd, cd, exp, ka)
        if kf is None:
            kf = kd
        self.reflective_bdrf = GlossySpecular(sampler, kf, cd, exp)

    def shade(self, sr):
        L = super(GlossyReflective, self).shade(sr)

        wo = -sr.ray.direction
        fr, wi, pdf = self.reflective_bdrf.sample_f(sr, wo)

        reflected_ray = Ray(sr.hit_point, wi)

        L += fr * sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1) * sr.normal.dot(wi) / pdf
        return L


class Emissive(Material):
    def __init__(self, ls, color):
        super(Emissive, self).__init__()
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
