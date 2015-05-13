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

    def render_progressive(self, world, pixel_func):
        vp = world.viewplane
        tracer = world.tracer

        pixel_size = vp.pixel_size
        resolution = vp.resolution

        coords = []
        color_data = []
        for row_el in vp:
            coords.append([])
            color_data.append([])
            for coord in row_el:
                samples = vp.sampler.sample()
                coords[-1].append(samples)
                color_data[-1].append(numpy.zeros(3))

        sample_to_hit = range(len(samples))
        random.shuffle(sample_to_hit)

        sample_count = 0
        for sample_idx in sample_to_hit:
            sample_count += 1
            print 'iteration ' + str(sample_count)
            for row, row_el in enumerate(coords):
                for col, samples in enumerate(row_el):
                    color = numpy.zeros(3)
                    sample = coords[row][col][sample_idx]
                    plane_point = numpy.zeros(3)
                    plane_point[0] = pixel_size * (col - resolution[0] / 2 + sample[0])
                    plane_point[1] = pixel_size * (row - resolution[1] / 2 + sample[1])
                    ray_dir = self.ray_direction(plane_point)
                    ray = Ray(self.eye, ray_dir)
                    color_hit = numpy.array(tracer.trace_ray(ray))
                    color_data[row][col] += color_hit ** 2.0

                    color = color_data[row][col]
                    n_color = color / sample_count
                    pixel_func((col, row), int(n_color[0]*255), int(n_color[1]*255), int(n_color[2]*255))


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

    def render_progressive(self, world, pixel_func):
        vp = world.viewplane
        tracer = world.tracer

        pixel_size = vp.pixel_size
        resolution = vp.resolution

        coords = []
        color_data = []
        for row_el in vp:
            coords.append([])
            color_data.append([])
            for coord in row_el:
                samples = vp.sampler.sample()
                disk_samples = vp.sampler.sample_unit_disk()
                coords[-1].append(zip(samples, disk_samples))
                color_data[-1].append(numpy.zeros(3))

        sample_to_hit = range(len(samples))
        random.shuffle(sample_to_hit)

        sample_count = 0
        for sample_idx in sample_to_hit:
            sample_count += 1
            print 'iteration ' + str(sample_count)
            for row, row_el in enumerate(coords):
                for col, samples in enumerate(row_el):
                    color = numpy.zeros(3)
                    sample, disk_sample = coords[row][col][sample_idx]
                    plane_point = numpy.zeros(3)
                    plane_point[0] = pixel_size*(col - resolution[0] / 2 + sample[0])
                    plane_point[1] = pixel_size*(row - resolution[1] / 2 + sample[1])

                    lens_point = self.eye + numpy.array([disk_sample[0], disk_sample[1], 0.0]) * self.lens_radius

                    ray_dir = self.ray_direction(plane_point, lens_point)
                    ray = Ray(lens_point, ray_dir)
                    color_hit = numpy.array(tracer.trace_ray(ray))
                    color_data[row][col] += color_hit ** 2.0

                    color = color_data[row][col]
                    n_color = color / sample_count
                    pixel_func((col, row), int(n_color[0]*255), int(n_color[1]*255), int(n_color[2]*255))

    def ray_direction(self, plane_point, lens_point):
        focal_plane_hitpoint = plane_point * self.focal_plane_distance / self.viewing_distance;
        direction = (focal_plane_hitpoint[0] - lens_point[0]) * self.u + (focal_plane_hitpoint[1] - lens_point[1]) * self.v - self.focal_plane_distance * self.w
        return direction / numpy.linalg.norm(direction)