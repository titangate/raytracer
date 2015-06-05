from sampler import RegularSampler, MultiJitteredSampler
from geometry import Plane, Triangle, Mesh
from tracer import ViewPlane, Tracer
from material import Matte
from light import AmbientLight, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
from mesh_parser import read_mesh


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'h'

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
            sampler = MultiJitteredSampler(sample_dim=2)

        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(5., 2., -3.), up=(0.,1.,0.), lookat=(0.,.5,0.), viewing_distance=700.)

        world.background_color = (0.0,0.0,0.0)
        world.tracer = Tracer(world)
        world.objects = []

        matte2 = Matte(1, 1, numpy.array((1.,1.,1.)))  # white

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .2)
        world.ambient_color = occluder

        import ipdb; ipdb.set_trace()
        mesh = read_mesh(open('meshes/mesh1.obj'))
        mesh.material = matte2
        world.objects.append(mesh)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((1., 2., 2.)), radius=4, attenuation=2, cast_shadow=False)
        ]
