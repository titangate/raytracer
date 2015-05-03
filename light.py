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
	def __init__(self, color, ls, location):
		self.color = color
		self.ls = ls
		self.location = location

	def get_direction(self, shader_rec):
		v = self.location - shader_rec.hit_point
		return v / numpy.linalg.norm(v)

	def L(self, shader_rec):
		return self.ls * self.color


class DirectionLight(object):
	def __init__(self, color, ls, direction):
		self.color = color
		self.ls = ls
		self.direction = direction

	def get_direction(self, shader_rec):
		return self.direction

	def L(self, shader_rec):
		return self.ls * self.color
