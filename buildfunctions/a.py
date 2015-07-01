from sampler import RegularSampler, MultiJitteredSampler
from camera import ThinLensCamera
from tracer import ViewPlane, Tracer
from light import AmbientOccluder, DirectionLight
from geometry import Plane, Sphere
from material import Phong, Matte
import numpy
from buildfunctionbase import BuildFunctionBase


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'a'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 40)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (320, 200)
            pixel_size = 1
            sampler = MultiJitteredSampler(sample_dim=7)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = ThinLensCamera(lens_radius=20.0, focal_plane_distance=500.0, eye=(0., -200., 600.), up=(0., -1., 0.), lookat=(0., -250., 0.), viewing_distance=200.)

        world.background_color = (0.0, 0.0, 0.0)
        world.tracer = Tracer(world)
        world.objects = []

        world.ambient_color = AmbientOccluder(numpy.array([0.2, 0.2, 0.2]), 1, sampler)

        world.lights = [
            DirectionLight(numpy.array([0,1,1]),1,numpy.array([0,-0.707,0.707]),True),
            DirectionLight(numpy.array([1,0,1]),1,numpy.array([0,-0.707,-0.707]),True),
            DirectionLight(numpy.array([1,1,0]),1,numpy.array([0.707,-0.707,0]),True),
        ]

        world.objects.append(Plane(origin=(0.0,25,0), normal=(0,-1,0), material=Matte(1,1,numpy.array([0.8,0.8,0.8]))))
        world.objects.append(Sphere(
                    center=(-300, -100, 100),
                    radius=100.0,
                    material=Phong(ka=1, kd=1, ks=1, exp=1, cd=numpy.array([0.8, 0.8, 0.8]))))
        world.objects.append(Sphere(
                    center=(-75, -100, -100),
                    radius=100.0,
                    material=Matte(ka=1, kd=1, cd=numpy.array([0.8, 0.8, 0.8]))))
        world.objects.append(Sphere(
                    center=(75, -100, 100),
                    radius=100.0,
                    material=Phong(ka=1, kd=1, ks=1, exp=1, cd=numpy.array([0.8, 0.8, 0.8]))))
        world.objects.append(Sphere(
                    center=(300, -100, -300),
                    radius=100.0,
                    material=Phong(ka=1, kd=1, ks=1, exp=100, cd=numpy.array([0.8, 0.8, 0.8]))))
