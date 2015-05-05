import numpy

class AmbientLight(object):
	def __init__(self, color, ls):
		self.color = color
		self.ls = ls

	def get_direction(self, shader_rec):
		return numpy.zeros(3)

	def L(self, shader_rec):
		return self.ls * self.color

class PointLight(object):
	def __init__(self, color, ls, location, attenuation=0, radius=300):
		self.color = color
		self.ls = ls
		self.location = location
		self.attenuation = attenuation
		self.radius = radius

	def get_direction(self, shader_rec):
		v = self.location - shader_rec.hit_point
		return v / numpy.linalg.norm(v)

	def L(self, shader_rec):
		color = self.ls * self.color
		if self.attenuation != 0:
			color *= (numpy.linalg.norm(shader_rec.hit_point - self.location) / self.radius) ** -self.attenuation

		return color


class DirectionLight(object):
	def __init__(self, color, ls, direction):
		self.color = color
		self.ls = ls
		self.direction = direction

	def get_direction(self, shader_rec):
		return self.direction

	def L(self, shader_rec):
		return self.ls * self.color
