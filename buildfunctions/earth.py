from sampler import RegularSampler, MultiJitteredSampler
from geometry import Plane, Sphere
from tracer import ViewPlane, Tracer
from material import Matte, SV_Matte
from texture import ImageTexture, SphericalMap
from light import AmbientOccluder
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'earth'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (300, 300)
            pixel_size = 3.2 / 3.
            sampler = MultiJitteredSampler(sample_dim=10)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(35., 10., 0.), up=(0.,1.,0.), lookat=(0.,1.,0.), viewing_distance=5000.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        mapping = SphericalMap()
        texture = ImageTexture(image='textures/earth_8k.jpg',
                               color=numpy.array((1., 1., 1.)),
                               mapping=mapping)
        matte1 = SV_Matte(ka=1., kd=0., cd=texture)  # yellow
        matte2 = Matte(1., 0, numpy.array((1.,1.,1.)))  # white

        occluder = AmbientOccluder(numpy.array((1.,1.,1.)), 1., sampler)
        world.ambient_color = occluder

        sphere = Sphere(center=numpy.array((0, 1, 0)), radius=1., material=matte1)
        world.objects.append(sphere)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte2)
        world.objects.append(plane)

        world.lights = []
