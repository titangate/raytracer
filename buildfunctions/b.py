from sampler import RegularSampler, MultiJitteredSampler
from geometry import Plane, Sphere
from tracer import ViewPlane, Tracer
from material import Matte
from light import AmbientOccluder
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'b'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (100, 100)
            pixel_size = 3.2
            sampler = MultiJitteredSampler(sample_dim=10)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(35., 10., 45.), up=(0.,1.,0.), lookat=(0.,1.,0.), viewing_distance=5000.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte1 = Matte(0.75, 0, numpy.array((1.,1.,0)))  # yellow
        matte2 = Matte(0.75, 0, numpy.array((1.,1.,1.)))  # white

        occluder = AmbientOccluder(numpy.array((1.,1.,1.)), 1., sampler)
        world.ambient_color = occluder

        sphere = Sphere(center=numpy.array((1.,1.,1.)), radius=1., material=matte1)
        world.objects.append(sphere)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte2)
        world.objects.append(plane)

        world.lights = []
