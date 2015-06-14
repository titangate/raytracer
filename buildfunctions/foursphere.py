from sampler import RegularSampler, MultiJitteredSampler
from tracer import ViewPlane, Tracer
from material import Matte, Reflective, Phong
from light import AmbientLight, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
from mesh_parser import read_mesh
from kdtree import BoundingBoxes, KDTree
from geometry import Plane, Sphere


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'foursphere'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (200, 200)
            pixel_size = .64 * 2.5
            sampler = MultiJitteredSampler(sample_dim=1)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(4., 0., 4.), up=(0.,1.,0.), lookat=(0.,0.,0.), viewing_distance=1300.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world, 20)
        world.objects = []

        mirror_mat = Reflective(0.6, numpy.array((1.,1.,1.)), 500)  # white
        mirror_mat_r = Reflective(0.6, numpy.array((1.,.1,.1)), 500)  # white
        mirror_mat_g = Reflective(0.6, numpy.array((.1,1.,.1)), 500)  # white
        mirror_mat_b = Reflective(0.6, numpy.array((.1,.1,1.)), 500)  # white

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .05)
        world.ambient_color = occluder

        radius = 1

        sphere1 = Sphere(center=numpy.array((1, 0, -1 / 2 ** 0.5)), radius=radius, material=mirror_mat_r)
        world.objects.append(sphere1)

        sphere2 = Sphere(center=numpy.array((-1, 0, -1 / 2 ** 0.5)), radius=radius, material=mirror_mat)
        world.objects.append(sphere2)

        sphere3 = Sphere(center=numpy.array((0, 1, 1 / 2 ** 0.5)), radius=radius, material=mirror_mat_g)
        world.objects.append(sphere3)

        sphere4 = Sphere(center=numpy.array((0, -1, 1 / 2 ** 0.5)), radius=radius, material=mirror_mat_b)
        world.objects.append(sphere4)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 0.4, numpy.array((0., 0., 0.)), radius=1, attenuation=2, cast_shadow=False)
        ]
