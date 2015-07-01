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

    BUILD_FUNCTION_NAME = 'mirroralt'

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
        world.camera = PinholeCamera(eye=(0., 2., -7.), up=(0.,1.,0.), lookat=(0.,1.5,0.), viewing_distance=300.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte1 = Phong(ka=1, kd=1, ks=1, exp=100, cd=numpy.array((1., .84, .1)))
        matte2 = Matte(ka=1, kd=1, cd=numpy.array([1., .84, 1.]))
        matte3 = Matte(ka=1, kd=1, cd=numpy.array([.2, .3, 1.]))
        mirror_mat = Reflective(ka=.6, kd=.6, ks=.6, kr=1.0, exp=500, cd=numpy.array((1., 1., 1.)))

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .2)
        world.ambient_color = occluder

        mesh = read_mesh(open('meshes/teapot.obj'))
        mesh.compute_smooth_normal()
        mesh.material = mirror_mat
        boxes = mesh.get_bounding_boxes()
        tree = KDTree(BoundingBoxes(boxes))
        world.objects.append(tree)

        sphere2 = Sphere(center=numpy.array((-2.5,0.5,-1.5)), radius=1., material=matte1)
        world.objects.append(sphere2)

        sphere3 = Sphere(center=numpy.array((3.5,1.5,2.5)), radius=2., material=matte2)
        world.objects.append(sphere3)

        sphere4 = Sphere(center=numpy.array((2.5,0.5,-1.5)), radius=.7, material=matte1)
        world.objects.append(sphere4)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=matte3)
        world.objects.append(plane)

        plane2 = Plane(origin=(0,0,8), normal=(0,0,-1), material=matte2)
        world.objects.append(plane2)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((1., 8., 2.)), radius=10, attenuation=2, cast_shadow=True)
        ]
