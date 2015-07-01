from sampler import RegularSampler, MultiJitteredSampler
from camera import PinholeCamera
from tracer import ViewPlane, AreaLightTracer
from light import AmbientOccluder, AmbientLight, EnvironmentLight
from geometry import Plane, Sphere, ConcaveSphere
from material import Phong, Matte, Emissive
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'd'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (500, 500)
            pixel_size = .64
            sampler = MultiJitteredSampler(sample_dim=1)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = AreaLightTracer(world)
        # world.tracer = Tracer(world)
        world.objects = []

        emissive = Emissive(1.5, numpy.array((1.,1.,1.)))

        world.objects = []

        concave_sphere = ConcaveSphere(numpy.array((0., 0., 0.)), 100000., emissive)
        concave_sphere.cast_shadow = False
        world.objects.append(concave_sphere)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        d = (1. / 3) ** 0.5 * 20
        world.camera = PinholeCamera(eye=(d, d, d), up=(0.,1.,0.), lookat=(0.,0.,0.), viewing_distance=200.)

        matte1 = Phong(ka=1, kd=1, ks=1, exp=1, cd=numpy.array([1., 1., 0]))
        matte2 = Matte(ka=1, kd=1, cd=numpy.array([1., 1., 1.]))

        occluder = AmbientOccluder(numpy.array((1.,0.,0.)), .5, sampler)
        occluder = AmbientLight(numpy.array((1.,1.,1.)), .0)
        world.ambient_color = occluder

        sphere = Sphere(center=numpy.array((0., 2.5, 5)), radius=5., material=matte1)
        world.objects.append(sphere)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte2)
        world.objects.append(plane)

        world.lights = [
            EnvironmentLight(emissive, sampler)
        ]
