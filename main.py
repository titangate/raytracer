from PIL import Image
import pygame
import math
import numpy
import sys
import random

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
        t = numpy.linalg.norm((self.origin - ray.origin) * self.normal / numpy.dot(ray.direction, self.normal))
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

class RegularSampler(object):
    def sample(self, row, column, resolution, pixel_size):
        origin = numpy.zeros(3)
        origin[0] = pixel_size*(column - resolution[0] / 2 + 0.5)
        origin[1] = pixel_size*(row - resolution[1] / 2 + 0.5)
        origin[2] = 1000.0
        return (origin,)

class MultiJitteredSampler(object):
    def __init__(self, sample_dim=2, pattern_size=83):
        self.sample_dim = sample_dim
        self.pattern_size = pattern_size
        self.patterns = []
        for i in xrange(pattern_size):
            samples = []
            idx_to_shuffle_row = range(sample_dim)
            idx_to_shuffle_col = range(sample_dim)
            random.shuffle(idx_to_shuffle_col)
            random.shuffle(idx_to_shuffle_row)
            dim = float(sample_dim)
            for i in xrange(sample_dim):
                for j in xrange(sample_dim):
                    samples.append(((i + random.uniform(0,1)) / dim ** 2 + idx_to_shuffle_row[i] / dim, (j + random.uniform(0,1)) / dim ** 2 + idx_to_shuffle_col[j] / dim))
            self.patterns.append(samples)

    def sample(self, row, column, resolution, pixel_size):
        rays = []
        for sample in random.choice(self.patterns):
            origin = numpy.zeros(3)
            origin[0] = pixel_size*(column - resolution[0] / 2 + sample[0] )
            origin[1] = pixel_size*(row - resolution[1] / 2 + sample[1] )
            origin[2] = 1000.0
            rays.append(origin)
        return rays

class ConcentricMapSampler(MultiJitteredSampler):
    def __init__(self, *args, **kwargs):
        super(ConcentricMapSampler, self).__init__(*args, **kwargs)
        concentric_patterns = []
        for samples in self.patterns:
            concentric_sample = []
            for sample in samples:
                x = sample[0] * 2 - 1
                y = sample[1] * 2 - 1
                if x > -y:
                    if x > y:
                        r = x
                        phi = y / x
                    else:
                        r = y
                        phi = 2 - x / y
                else:
                    if x < y:
                        r = -x
                        phi = 4 + y / x
                    else:
                        r = -y
                        if y != 0:
                            # at origin
                            phi = 6 - x / y
                        else:
                            phi = 0
                phi *= numpy.pi / 4.0
                concentric_sample.append((r * numpy.cos(phi), r * numpy.sin(phi)))
            concentric_patterns.append(concentric_sample)
        self.patterns = concentric_patterns


class ViewPlane(object):
    def __init__(self, resolution, pixel_size, sampler=RegularSampler):
        self.resolution = resolution
        self.pixel_size = pixel_size
        self.sampler = sampler


    def iter_row(self, row):
        for column in xrange(self.resolution[0]):
            rays = [Ray(origin=origin, direction=(0,0,-1.0)) for origin in self.sampler.sample(row, column, self.resolution, self.pixel_size)]
            yield (rays, (column,row))

    def __iter__(self):
        for row in xrange(self.resolution[1]):
            yield self.iter_row(row) 

class World(object):
    def __init__(self):
        self.viewplane = ViewPlane(resolution=(320,200), pixel_size=1.0, sampler=MultiJitteredSampler(sample_dim=2))
        self.background_color = (0.0,0.0,0.0)
        self.objects = []
        # initiate objects
        self.objects.append(Sphere(center=(0.0,0.0,0.0), radius=85.0, color=(1.0,0,0)))
        self.objects.append(Sphere(center=(50.0,10.0,30.0), radius=85.0, color=(1.0,1.0,0)))
        self.objects.append(Plane(origin=(0.0,0.0,-10.0), normal=(0,0,1.0), color=(0,0,1.0)))

    def hit_bare_bones_objects(self, ray):
        tmin = INF
        hit = None
        for obj in self.objects:
            shader_rec = obj.hit(ray)
            if shader_rec and shader_rec.tmin < tmin:
                hit = shader_rec
                tmin = shader_rec.tmin
                shader_rec.color = obj.get_color()
        return hit

    def render(self):
        pygame.init() 
        window = pygame.display.set_mode(self.viewplane.resolution) 
        pxarray = pygame.PixelArray(window)
        im = Image.new("RGB", self.viewplane.resolution)
        tracer = Tracer(self)
        for row in self.viewplane:
            for rays, pixel in row:
                color = numpy.zeros(3)
                for ray in rays:
                    color += tracer.trace_ray(ray)
                color /= len(rays)
                im.putpixel(pixel, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
                pxarray[pixel[0]][pixel[1]] = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            pygame.display.flip() 

        im.save("render.png", "PNG")
        while True: 
           for event in pygame.event.get(): 
              if event.type == pygame.QUIT: 
                  sys.exit(0)

if __name__ == "__main__":
    w=World()
    w.render()
