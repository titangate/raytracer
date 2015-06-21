from sampler import RegularSampler, MultiJitteredSampler
from tracer import ViewPlane, Tracer
from material import Matte
from light import AmbientLight, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
from mesh_parser import read_mesh, read_mesh_complex
from kdtree import BoundingBoxes, KDTree
from geometry import Plane


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'cornellbox'

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
            sampler = MultiJitteredSampler(sample_dim=1)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(0., 1, 6.),
            up=(0.,1.,0.), lookat=(0.,1,0.), viewing_distance=800.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .2)
        world.ambient_color = occluder

        world_objects = read_mesh_complex('CornellBox/CornellBox-Original.obj')

        boxes = []
        for key, mesh in world_objects.iteritems():
            mesh.compute_normal()
            boxes += mesh.get_bounding_boxes()

        tree = KDTree(BoundingBoxes(boxes))
        tree.print_tree()
        world.objects.append(tree)

        world.lights = [
            PointLight(numpy.array((1., 1., 1.)), 1., numpy.array((0., 1., 4.)), radius=4, attenuation=2, cast_shadow=True)
        ]
