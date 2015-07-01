from sampler import RegularSampler, MultiJitteredSampler
from geometry import Plane, AxisAlignedBox
from tracer import ViewPlane, Tracer
from material import Matte
from light import AmbientOccluder, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'e'

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

        matte1 = Matte(ka=.75, kd=1, cd=numpy.array([1., 1., 0]))
        matte2 = Matte(ka=.75, kd=1, cd=numpy.array([1., 1., 1.]))

        occluder = AmbientOccluder(numpy.array((1.,1.,1.)), .2, sampler)
        world.ambient_color = occluder

        box = AxisAlignedBox(-1, 1, 0.5, 2.5, -1, 1, material=matte2)
        world.objects.append(box)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte1)
        world.objects.append(plane)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((2., 4., -2.)), radius=5, attenuation=2)
        ]
