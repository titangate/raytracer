import numpy
from tracer import Ray


class Material(object):
    def __init__(self):
        self.receives_shadow = True

    def get_Le(self):
        return numpy.array((0.,0.,0.))


class BDRF(object):
    pass


class SV_Lambertian(BDRF):
    def __init__(self, sampler, kd, cd):
        super(SV_Lambertian, self).__init__()
        self.sampler = sampler
        self.kd = kd
        self.cd = cd

    def rho(self, sr, wo):
        return self.kd * self.cd.get_color(sr)

    def f(self, sr, wo, wi):
        return self.kd * self.cd.get_color(sr) / numpy.pi


class Lambertian(BDRF):
    def __init__(self, sampler, kd, cd):
        super(Lambertian, self).__init__()
        self.sampler = sampler
        self.kd = kd
        self.cd = cd

    def f(self, shader_rec, wo, wi):
        return self.kd * self.cd / numpy.pi

    def sample_f(self, shader_rec, wo):
        w = shader_rec.normal
        v = numpy.cross(numpy.array([0.0034, 1, 0.0071]), w)
        v /= numpy.linalg.norm(v)
        u = numpy.cross(v, w)

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

        pdf = shader_rec.normal.dot(wi)
        return self.kd * self.cd / pdf, wi, pdf

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


class BTDF(object):
    def __init__(self, sampler=None):
        self.sampler = sampler


class PerfectTransmitter(BTDF):
    def __init__(self, ior, kt, sampler=None):
        super(PerfectTransmitter, self).__init__(sampler)
        self.ior = ior
        self.kt = kt

    def total_internal_reflection(self, sr):
        wo = -sr.ray.direction
        cos_thetai = sr.normal.dot(wo)
        eta = self.ior
        if cos_thetai < 0.:
            eta = 1.0 / eta

        return 1. - (1. - cos_thetai ** 2) / eta ** 2 < 0.

    def sample_f(self, sr, wo):
        n = sr.normal
        cos_thetai = n.dot(wo)
        eta = self.ior

        if cos_thetai < 0.:
            cos_thetai = -cos_thetai
            n = -n
            eta = 1.0 / eta

        temp = 1. - (1. - cos_thetai ** 2) / eta ** 2
        cos_theta2 = temp ** 0.5

        wt = -wo / eta - (cos_theta2 - cos_thetai / eta) * n

        return self.kt / eta ** 2 / abs(sr.normal.dot(wt)), wt


class FresnelTransmitter(BTDF):
    def __init__(self, sampler, kd, cd, eta_in, eta_out):
        super(FresnelTransmitter, self).__init__(sampler)
        self.eta_in = eta_in
        self.eta_out = eta_out
        self.kd = kd
        self.cd = cd

    def total_internal_reflection(self, sr):
        normal = sr.normal
        cos_thetai = -normal.dot(sr.ray.direction)

        if cos_thetai < 0.:
            eta = self.eta_out / self.eta_in
        else:
            eta = self.eta_in / self.eta_out

        return 1. - (1. - cos_thetai ** 2) / eta ** 2 < 0.

    def sample_f(self, sr, wo, fresnel):
        normal = sr.normal
        cos_thetai = -normal.dot(sr.ray.direction)

        if cos_thetai < 0.:
            normal = -normal
            eta = self.eta_out / self.eta_in
            cos_thetai = -cos_thetai
        else:
            eta = self.eta_in / self.eta_out

        temp = 1. - (1. - cos_thetai ** 2) / eta ** 2
        cos_theta2 = temp ** 0.5

        wt = -wo / eta - (cos_theta2 - cos_thetai / eta) * normal

        return (1 - fresnel) / eta ** 2 / abs(sr.normal.dot(wt)), wt


class FresnelReflector(PerfectSpecular):
    def __init__(self, sampler, kd, cd, eta_in, eta_out):
        super(FresnelReflector, self).__init__(sampler, kd, cd)
        self.eta_in = eta_in
        self.eta_out = eta_out

    def fresnel(self, sr):
        normal = sr.normal
        ndotd = -normal.dot(sr.ray.direction)

        if ndotd < 0.:
            normal = -normal
            eta = self.eta_out / self.eta_in
        else:
            eta = self.eta_in / self.eta_out

        cos_thetai = -normal.dot(sr.ray.direction)
        temp = 1. - (1. - cos_thetai ** 2) / eta ** 2
        cos_thetat = temp ** 0.5
        r_parallel = (eta * cos_thetai - cos_thetat) / (eta * cos_thetai + cos_thetat)
        r_perpendicular = (cos_thetai - eta * cos_thetat) / (cos_thetai + eta * cos_thetat)
        kr = .5 * (r_parallel ** 2 + r_perpendicular ** 2)

        return kr

    def sample_f(self, sr, wo, fresnel):
        ndotwo = sr.normal.dot(wo)
        wi = -wo + 2.0 * sr.normal * ndotwo

        return fresnel / abs(sr.normal.dot(wi)), wi


class Matte(Material):
    def __init__(self, ka, kd, cd, sampler=None):
        super(Matte, self).__init__()
        self.ambient_brdf = Lambertian(None, ka, cd)
        self.diffuse_brdf = Lambertian(sampler, kd, cd)

    def set_ka(self, ka):
        self.ambient_brdf.kd = ka

    def set_kd(self, kd):
        self.diffuse_brdf.kd = kd

    def set_cd(self, c):
        self.ambient_brdf.cd = c
        self.diffuse_brdf.cd = c

    def sample_f(self, shader_rec, wo):
        return self.diffuse_brdf.sample_f(shader_rec, wo)

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

    def path_shade(self, sr):
        f, wi, pdf = self.diffuse_brdf.sample_f(sr, -sr.ray.direction)
        ndotwi = sr.normal.dot(wi)

        reflected_ray = Ray(sr.hit_point, wi)

        return f * sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1) * ndotwi / pdf


class SV_Matte(Material):
    def __init__(self, ka, kd, cd, sampler=None):
        super(SV_Matte, self).__init__()
        self.ambient_brdf = SV_Lambertian(None, ka, cd)
        self.diffuse_brdf = SV_Lambertian(sampler, kd, cd)

    def set_ka(self, ka):
        self.ambient_brdf.kd = ka

    def set_kd(self, kd):
        self.diffuse_brdf.kd = kd

    def set_cd(self, c):
        self.ambient_brdf.cd = c
        self.diffuse_brdf.cd = c

    def sample_f(self, shader_rec, wo):
        return self.diffuse_brdf.sample_f(shader_rec, wo)

    def shade(self, sr):
        wo = -sr.ray.direction
        L = self.ambient_brdf.rho(sr, wo) * sr.world.ambient_color.L(sr)
        for light in sr.world.lights:
            wi = light.get_direction(sr)
            ndotwi = sr.normal.dot(wi)
            ndotwo = sr.normal.dot(wo)

            if ndotwi > 0 and ndotwo > 0.:
                in_shadow = False
                if light.cast_shadow:
                    shadow_ray = Ray(sr.hit_point, wi)
                    in_shadow = light.in_shadow(shadow_ray, sr)
                if not in_shadow:
                    L += self.diffuse_brdf.f(sr, wo, wi) * light.L(sr) * ndotwi
        return L


class Phong(Material):
    def __init__(self, ka, kd, ks, cd, exp, sampler=None):
        super(Phong, self).__init__()
        self.ambient_brdf = Lambertian(sampler, ka, cd)
        self.diffuse_brdf = Lambertian(sampler, kd, cd)
        self.specular_brdf = GlossySpecular(sampler, ks, cd, exp)

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
    def __init__(self, ka, kd, ks, kr, cd, exp, sampler=None):
        super(Reflective, self).__init__(ka, kd, ks, cd, exp, sampler)
        self.reflective_bdrf = PerfectSpecular(None, kr, cd)

    def shade(self, sr):
        L = super(Reflective, self).shade(sr)

        wo = -sr.ray.direction
        fr, wi, pdf = self.reflective_bdrf.sample_f(sr, wo)

        reflected_ray = Ray(sr.hit_point, wi)

        L += fr * sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1) * sr.normal.dot(wi)
        return L


class GlossyReflective(Phong):
    def __init__(self, ka, kd, ks, kr, cd, exp, sampler):
        super(GlossyReflective, self).__init__(ka, kd, ks, cd, exp, sampler)
        self.reflective_bdrf = GlossySpecular(sampler, kr, cd, exp)

    def shade(self, sr):
        L = super(GlossyReflective, self).shade(sr)

        wo = -sr.ray.direction
        fr, wi, pdf = self.reflective_bdrf.sample_f(sr, wo)

        reflected_ray = Ray(sr.hit_point, wi)

        L += fr * sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1) * sr.normal.dot(wi) / pdf
        return L


class Transparent(Phong):
    def __init__(self, ka, kd, ks, cd, exp, sampler, ior, kt, kr):
        super(Transparent, self).__init__(ka, kd, ks, cd, exp, sampler)
        self.reflective_bdrf = PerfectSpecular(sampler, kr, cd)
        self.specular_btrf = PerfectTransmitter(ior, kt)

    def shade(self, sr):
        L = super(Transparent, self).shade(sr)

        wo = -sr.ray.direction
        fr, wi, pdf = self.reflective_bdrf.sample_f(sr, wo)

        reflected_ray = Ray(sr.hit_point, wi)

        reflected_component = sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1)

        if self.specular_btrf.total_internal_reflection(sr):
            L += reflected_component
        else:
            ft, wt = self.specular_btrf.sample_f(sr, wo)
            transmitted_ray = Ray(sr.hit_point, wt)
            L += reflected_component * sr.normal.dot(wi) * fr
            L += ft * sr.world.tracer.trace_ray(transmitted_ray, sr.depth + 1) * abs(sr.normal.dot(wt))
        return L


class Dielectric(Phong):
    def __init__(self, ka, kd, ks, cd, exp, sampler, eta_in, eta_out, cf_in, cf_out):
        super(Dielectric, self).__init__(ka, kd, ks, cd, exp, sampler)
        self.fresnel_brdf = FresnelReflector(sampler, kd, cd, eta_in, eta_out)
        self.fresnel_btdf = FresnelTransmitter(sampler, kd, cd, eta_in, eta_out)
        self.cf_in = cf_in
        self.cf_out = cf_out

    def shade(self, sr):
        L = super(Dielectric, self).shade(sr)

        wo = -sr.ray.direction
        fresnel = self.fresnel_brdf.fresnel(sr)
        fr, wi = self.fresnel_brdf.sample_f(sr, wo, fresnel)
        reflected_ray = Ray(sr.hit_point, wi)
        ndotwi = sr.normal.dot(wi)

        tr_list = [0]
        if self.fresnel_btdf.total_internal_reflection(sr):
            Lr = sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1, tr_list)
            t = tr_list[0]
            if ndotwi < 0.:
                # reflected ray is inside
                L += Lr * self.cf_in ** t
            else:
                # reflected ray is outside
                L += Lr * self.cf_out ** t
        else:
            ft, wt = self.fresnel_btdf.sample_f(sr, wo, fresnel)
            transmitted_ray = Ray(sr.hit_point, wt)
            ndotwt = sr.normal.dot(wt)

            tt_list = [0]
            Lr = sr.world.tracer.trace_ray(reflected_ray, sr.depth + 1, tr_list)
            Lt = sr.world.tracer.trace_ray(transmitted_ray, sr.depth + 1, tt_list)
            tt = tt_list[0]
            tr = tr_list[0]
            if ndotwi < 0.:
                L += fr * Lr * self.cf_in ** tr * abs(ndotwi)
                L += ft * Lt * self.cf_out ** tt * abs(ndotwt)
            else:
                L += fr * Lr * self.cf_out ** tr * abs(ndotwi)
                L += ft * Lt * self.cf_in ** tt * abs(ndotwt)
        return L


class Emissive(Material):
    def __init__(self, ls, color):
        super(Emissive, self).__init__()
        self.ls = ls
        self.color = color

    def get_Le(self, shader_rec):
        return self.ls * self.color

    def shade(self, shader_rec):
        return numpy.array((1., 1., 1.))

    def area_light_shade(self, shader_rec):
        if -shader_rec.normal.dot(shader_rec.ray.direction) > 0:
            return self.ls * self.color
        else:
            return numpy.array((0. ,0. ,0.))

    def path_shade(self, sr):
        ndotwi = sr.normal.dot(sr.ray.direction)
        if ndotwi < 0:
            return self.ls * self.color
        return numpy.array((0., 0., 0.))
