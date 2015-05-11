import numpy
from tracer import *


class PinholeCamera(object):
    def __init__(self, eye, lookat, up, viewing_distance):
        self.eye = numpy.array(eye)
        self.lookat = numpy.array(lookat)
        self.up = numpy.array(up)
        self.viewing_distance = viewing_distance
        self.compute_uwv()

    def compute_uwv(self):
        w = self.eye - self.lookat
        w /= numpy.linalg.norm(w)
        u = numpy.cross(w, self.up)
        u /= numpy.linalg.norm(u)
        v = numpy.cross(w, u)

        self.u = u
        self.v = v
        self.w = w

    def ray_direction(self, point):
        direction = point[0] * self.u + point[1] * self.v - self.viewing_distance * self.w
        return direction / numpy.linalg.norm(direction)

    def render(self, world, pixel_func):
        vp = world.viewplane
        tracer = world.tracer

        pixel_size = vp.pixel_size
        resolution = vp.resolution

        for row in vp:
            for coord in row:
                column, row = coord
                color = numpy.zeros(3)
                samples = vp.sampler.sample()
                for sample in samples:
                    plane_point = numpy.zeros(3)
                    plane_point[0] = pixel_size*(column - resolution[0] / 2 + sample[0])
                    plane_point[1] = pixel_size*(row - resolution[1] / 2 + sample[1])

                    ray_dir = self.ray_direction(plane_point)
                    ray = Ray(self.eye, ray_dir)
                    color += numpy.array(tracer.trace_ray(ray)) ** 2.0
                color /= len(samples)
                color ** 0.5
                pixel_func(coord, int(color[0]*255), int(color[1]*255), int(color[2]*255))


class ThinLensCamera(PinholeCamera):
    def __init__(self, lens_radius, focal_plane_distance, *args, **kwargs):
        self.lens_radius = lens_radius
        self.focal_plane_distance = focal_plane_distance
        super(ThinLensCamera, self).__init__(*args, **kwargs)

    def render(self, world, pixel_func):
        vp = world.viewplane
        tracer = world.tracer

        pixel_size = vp.pixel_size
        resolution = vp.resolution

        for row in vp:
            for coord in row:
                column, row = coord
                color = numpy.zeros(3)
                samples = vp.sampler.sample()
                disk_samples = vp.sampler.sample_unit_disk()
                for i, sample in enumerate(samples):
                    disk_sample = disk_samples[i]

                    plane_point = numpy.zeros(3)
                    plane_point[0] = pixel_size*(column - resolution[0] / 2 + sample[0])
                    plane_point[1] = pixel_size*(row - resolution[1] / 2 + sample[1])

                    lens_point = self.eye + numpy.array([disk_sample[0], disk_sample[1], 0.0]) * self.lens_radius

                    ray_dir = self.ray_direction(plane_point, lens_point)
                    ray = Ray(lens_point, ray_dir)
                    color += numpy.array(tracer.trace_ray(ray)) ** 2.0
                color /= len(samples)
                color ** 0.5
                pixel_func(coord, int(color[0]*255), int(color[1]*255), int(color[2]*255))

    def ray_direction(self, plane_point, lens_point):
        focal_plane_hitpoint = plane_point * self.focal_plane_distance / self.viewing_distance;
        direction = (focal_plane_hitpoint[0] - lens_point[0]) * self.u + (focal_plane_hitpoint[1] - lens_point[1]) * self.v - self.focal_plane_distance * self.w
        return direction / numpy.linalg.norm(direction)