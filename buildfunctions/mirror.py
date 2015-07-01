from sampler import RegularSampler, MultiJitteredSampler
from tracer import ViewPlane, Tracer
from material import Matte, Reflective
from light import AmbientLight, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
from mesh_parser import read_mesh
from kdtree import BoundingBoxes, KDTree
from geometry import Plane


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'mirror'

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

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(0., 2., 7.), up=(0.,1.,0.), lookat=(0.,1.5,0.), viewing_distance=300.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte1 = Matte(ka=1, kd=1, cd=numpy.array((1., .84, .1)))
        mirror_mat = Reflective(ka=0.6, kd=0.6, ks=0.6, kr=1.0, exp=100, cd=numpy.array((1., 1., 1.)))

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .2)
        world.ambient_color = occluder

        mesh = read_mesh(open('meshes/teapot.obj'))
        mesh.compute_smooth_normal()
        mesh.material = matte1
        boxes = mesh.get_bounding_boxes()
        tree = KDTree(BoundingBoxes(boxes))
        tree.print_tree()
        world.objects.append(tree)

        plane = Plane(origin=(0,0,0), normal=(0,1,0), material=mirror_mat)
        world.objects.append(plane)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((1., 8., 2.)), radius=10, attenuation=2, cast_shadow=True)
        ]
