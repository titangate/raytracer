from sampler import RegularSampler, MultiJitteredSampler
from tracer import ViewPlane, Tracer
from material import Matte, Reflective, Phong, GlossyReflective
from light import AmbientLight, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
from mesh_parser import read_mesh
from kdtree import BoundingBoxes, KDTree
from geometry import Plane, Sphere, CheckerPlane


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'glossyreflectivehall'

    @classmethod
    def build_function(cls, world, viewmode):
        bdrf_e = 400
        world.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
            sampler_bdrf = RegularSampler()
        else:
            resolution = (400, 400)
            pixel_size = .64 * 1.25
            sampler = MultiJitteredSampler(sample_dim=3)
            sampler_bdrf = MultiJitteredSampler(sample_dim=3,e=bdrf_e)
        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(2., 2., -4.), up=(0.,1.,0.), lookat=(0.,1.5,0.), viewing_distance=300.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte1 = Phong(ka=1, kd=1, ks=1, exp=100, cd=numpy.array((1., .84, .1)))
        matte2 = Matte(ka=1, kd=1, cd=numpy.array([1., .84, 1.]))
        matte3 = Matte(ka=1, kd=1, cd=numpy.array([1., 1., 1.]))
        mirror_mat = GlossyReflective(ka=0., kd=0., ks=0., kr=1.0, exp=bdrf_e, cd=numpy.array((1., 1., 1.)), sampler=sampler_bdrf)
        mirror_mat_alt = GlossyReflective(ka=0., kd=0., ks=0., kr=1.0, exp=10, cd=numpy.array((1., 1., 1.)), sampler=sampler_bdrf)

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .2)
        world.ambient_color = occluder

        sphere1 = Sphere(center=numpy.array((0,1,0)), radius=1., material=matte1)
        world.objects.append(sphere1)

        plane = CheckerPlane(origin=(0,0,0), normal=(0,1,0), material=matte3, alt_material=matte2,
                             up=(1,0,0), grid_size=0.5)
        world.objects.append(plane)

        plane2 = Plane(origin=(0,0,2.5), normal=(0,0,-1), material=mirror_mat)
        world.objects.append(plane2)

        plane3 = Plane(origin=(0,0,-4), normal=(0,0,1), material=mirror_mat_alt)
        world.objects.append(plane3)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((1., 8., 2.)), radius=10, attenuation=2, cast_shadow=True)
        ]
