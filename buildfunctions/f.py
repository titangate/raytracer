from sampler import RegularSampler, MultiJitteredSampler
from geometry import Plane, Triangle, AxisAlignedBox
from tracer import ViewPlane, Tracer
from material import Matte
from light import AmbientOccluder, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'f'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (200, 200)
            pixel_size = 1.6
            sampler = MultiJitteredSampler(sample_dim=3)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(35., 10., 45.), up=(0.,1.,0.), lookat=(0.,1.,0.), viewing_distance=5000.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte1 = Matte(0.75, 1, numpy.array((1.,1.,0)))  # yellow
        matte2 = Matte(0.75, 1, numpy.array((1.,1.,1.)))  # white

        occluder = AmbientOccluder(numpy.array((1.,1.,1.)), .2, sampler)
        world.ambient_color = occluder

        v0 = numpy.array((0.,1.,0.))
        v1 = numpy.array((0.,1.,2.))
        v2 = numpy.array((0.,3.,0.))

        triangle = Triangle(v0, v1, v2, material=matte2)
        world.objects.append(triangle)

        box = AxisAlignedBox(0, 10., -0.2, 0.2, -0.2, 0.2, material=matte2)
        world.objects.append(box)

        box = AxisAlignedBox(-0.2, 0.2, -0.2, 0.2, 0, 10., material=matte1)
        world.objects.append(box)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte1)
        world.objects.append(plane)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((2., 4., -2.)), radius=5, attenuation=2)
        ]
