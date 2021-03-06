from sampler import RegularSampler, MultiJitteredSampler
from camera import PinholeCamera
from tracer import ViewPlane, AreaLightTracer
from light import AreaLight, AmbientLight
from geometry import Plane, Sphere, Rectangle
from material import Phong, Matte, Emissive
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'c'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (400, 400)
            pixel_size = 0.8
            sampler = MultiJitteredSampler(sample_dim=7)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = AreaLightTracer(world)
        world.objects = []

        emissive = Emissive(4.8, numpy.array((1.,1.,1.)))

        world.objects = []

        rectangle = Rectangle(numpy.array((0., 10., -20.)), 4, 4, numpy.array((0., 0., 1.)), numpy.array((0., 1., 0.)), emissive, sampler)
        world.objects.append(rectangle)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        d = (1. / 3) ** 0.5 * 20
        world.camera = PinholeCamera(eye=(d, d, d), up=(0.,1.,0.), lookat=(0.,0.,0.), viewing_distance=200.)

        matte1 = Phong(ka=1, kd=1, ks=1, exp=1, cd=numpy.array([1., 1., 0]))
        matte2 = Matte(ka=1, kd=1, cd=numpy.array([1., 1., 1.]))

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .0)
        world.ambient_color = occluder

        sphere = Sphere(center=numpy.array((0., 2.5, 5)), radius=5., material=matte1)
        world.objects.append(sphere)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte2)
        world.objects.append(plane)

        world.lights = [
            AreaLight(emissive, rectangle, 5.)
        ]
