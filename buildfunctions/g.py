from sampler import RegularSampler, MultiJitteredSampler
from geometry import Plane, Sphere, Instance
from tracer import ViewPlane, Tracer
from material import Matte, Phong
from light import AmbientOccluder, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
import affinetransform as transform


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'g'

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
            sampler = MultiJitteredSampler(sample_dim=7)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(15., 5., 0.), up=(0.,1.,0.), lookat=(0.,1.,0.), viewing_distance=1000.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte1 = Phong(ka=1, kd=1, ks=1, exp=100, cd=numpy.array([1., 1., 0]))
        matte2 = Matte(ka=1, kd=1, cd=numpy.array([1., 1., 1.]))

        occluder = AmbientOccluder(numpy.array((1.,1.,1.)), .5, sampler)
        world.ambient_color = occluder

        sphere = Sphere(center=numpy.array((0.,0.,0.)), radius=1., material=matte1)
        mat = transform.scale(1.)
        mat = mat.dot(transform.translate(0, 1., 0.))
        mat = mat.dot(transform.rotate(numpy.array((0, 0.707, 0.707)), 2.))
        mat = mat.dot(transform.scale(1., 1, 2.))
        instance = Instance(sphere, mat)
        world.objects.append(instance)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte2)
        world.objects.append(plane)

        world.lights = [PointLight(numpy.array((0.,1.,1.)), 1., numpy.array((2., 4., -2.)), radius=5, attenuation=2)]
