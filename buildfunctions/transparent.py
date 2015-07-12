from sampler import RegularSampler, MultiJitteredSampler
from tracer import ViewPlane, WhittedTracer
from material import Matte, Reflective, Phong, Dielectric, Transparent
from light import AmbientLight, PointLight
from camera import PinholeCamera
import numpy
from buildfunctionbase import BuildFunctionBase
from mesh_parser import read_mesh
from kdtree import BoundingBoxes, KDTree
from geometry import Plane, Sphere, CheckerPlane, AxisAlignedBox


class BuildFunction(BuildFunctionBase):

    BUILD_FUNCTION_NAME = 'transparent'

    @classmethod
    def build_function(cls, world, viewmode):
        world.viewmode = viewmode
        bdrf_e = 100
        if viewmode == "realtime":
            resolution = (64, 64)
            pixel_size = 5
            sampler = RegularSampler()
            sampler_bdrf = RegularSampler()
        else:
            resolution = (400, 400)
            pixel_size = .64 * 1.25
            sampler = MultiJitteredSampler(sample_dim=10)
            sampler_bdrf = MultiJitteredSampler(sample_dim=10, e=bdrf_e)
        world.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size, sampler=sampler)
        world.camera = PinholeCamera(eye=(0, 8., -0), up=(1.,0.,0.), lookat=(0.,1.5,0.), viewing_distance=300.)

        world.background_color = numpy.array((0.2, 0.2, 0.2))
        world.tracer = WhittedTracer(world, 5)
        world.objects = []

        matte1 = Phong(ka=1, kd=1, ks=1, exp=100, cd=numpy.array((1., .84, .1)))
        matte2 = Matte(ka=1, kd=1, cd=numpy.array([1., .84, 1.]))
        matte3 = Matte(ka=1, kd=1, cd=numpy.array([1., 1., 1.]))
        matte4 = Matte(ka=1, kd=1, cd=numpy.array([.2, .3, 1.]))

        transparent_mat = Transparent(sampler=sampler_bdrf,
                                      ior=1.05,
                                      ka=0,
                                      kd=0,
                                      ks=0,
                                      kr=.1,
                                      cd=numpy.array((1., 1., 1.)),
                                      kt=.9,
                                      exp=0.)

        dielectric_mat = Dielectric(sampler=sampler_bdrf,
                                    ka=0,
                                    kd=0,
                                    ks=0,
                                    cd=numpy.array((1., 1., 1.)),
                                    exp=0,
                                    eta_in=0.8,
                                    eta_out=1,
                                    cf_in=numpy.array((.9, .6, .5)),
                                    cf_out=numpy.array((.9, .6, .5)))

        occluder = AmbientLight(numpy.array((1.,1.,1.)), .2)
        world.ambient_color = occluder

        # sphere1 = Sphere(center=numpy.array((0,2.1,-0.5)), radius=2., material=dielectric_mat)
        # sphere1.cast_shadow = False
        # world.objects.append(sphere1)

        mesh = read_mesh(open('meshes/teapot.obj'))
        mesh.compute_smooth_normal()
        mesh.cast_shadow = False
        mesh.material = dielectric_mat
        boxes = mesh.get_bounding_boxes()
        tree = KDTree(BoundingBoxes(boxes))
        world.objects.append(tree)
        # box1 = AxisAlignedBox(-1., 1., 0.05, 5., -1., 1., material=transparent_mat)
        # world.objects.append(box1)

        sphere2 = Sphere(center=numpy.array((-2.5,0.5,-2.5)), radius=2., material=matte1)
        world.objects.append(sphere2)

        # sphere3 = Sphere(center=numpy.array((-5,1.5,5)), radius=1., material=matte4)
        # world.objects.append(sphere3)

        plane = CheckerPlane(origin=(0,0,0), normal=(0,1,0), material=matte3, alt_material=matte2,
                             up=(1,0,0), grid_size=0.5)
        world.objects.append(plane)

        # plane2 = Plane(origin=(0,0,8), normal=(0,0,-1), material=matte2)
        # world.objects.append(plane2)

        world.lights = [
            PointLight(numpy.array((1.,1.,1.)), 1., numpy.array((1., 8., 2.)), radius=10, attenuation=2, cast_shadow=True)
        ]
