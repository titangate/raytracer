import numpy

class Lambertian(object):
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

class Matte(object):
	def __init__(self, kd, cd):
		self.ambient_brdf = Lambertian(None, kd, cd)
		self.diffuse_brdf = Lambertian(None, kd, cd)

	def set_ka(self, ka):
		self.ambient_brdf.set_kd(ka)

	def set_kd(self, kd):
		self.diffuse_brdf.set_kd(kd)

	def set_cd(self, c):
		self.ambient_brdf.set_cd(c)
		self.diffuse_brdf.set_cd(c)

	def shade(self, sr):
		wo = -sr.ray.direction
		L = self.ambient_brdf.rho(sr, wo) * sr.world.ambient_color.L(sr)
		for light in sr.world.lights:
			wi = light.get_direction(sr)
			ndotwi = sr.normal.dot(wi)

			if ndotwi > 0:
				L += self.diffuse_brdf.f(sr, wo, wi) * light.L(sr) * ndotwi
		return L