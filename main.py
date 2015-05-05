from PIL import Image
import pygame
import math
import numpy
import sys
import random
from sampler import *
from camera import PinholeCamera, ThinLensCamera
from tracer import *
from light import *

import argparse

import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    theta = numpy.asarray(theta)
    axis = axis/math.sqrt(numpy.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return numpy.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class World(object):
    def __init__(self, viewmode="realtime"):
        self.viewmode = viewmode
        if viewmode == "realtime":
            resolution = (64,40)
            pixel_size = 5
            sampler = RegularSampler()
        else:
            resolution = (320,200)
            pixel_size = 1
            sampler = MultiJitteredSampler(sample_dim=3)
        self.viewplane = ViewPlane(resolution=resolution, pixel_size=pixel_size,
            sampler=sampler)
        self.camera = ThinLensCamera(lens_radius = 10.0, focal_plane_distance=500.0, eye=(0.,0.,800.), up=(0.,1.,0.), lookat=(0.,0.,0.), viewing_distance=200.)
        self.background_color = (0.0,0.0,0.0)
        self.tracer = Tracer(self)
        self.objects = []

        self.ambient_color = AmbientLight(numpy.array([0.2,0.2,0.2]), 1)


        # initiate objects
        # for x in xrange(3):
        #     for y in xrange(3):
        #         color = numpy.array([x / 3., y / 3., .5])
        #         self.objects.append(Sphere(
        #             center=(x * 250 - 250.,y * 120 - 150., (x * 3+y) * 40 + 250),
        #             radius=50.0,
        #             material=Matte(1,color)))
        self.lights = [
            DirectionLight(numpy.array([0,1,1]),1,numpy.array([0,1,0])),
            DirectionLight(numpy.array([1,0,0]),1,numpy.array([0,0.707,0.707])),
            #PointLight(numpy.array([1,0,0]),1,numpy.array([0,-200,300]),2,300),
        ]
        #self.objects.append(Sphere(center=(50.0,10.0,500.0), radius=85.0, color=(1.0,1.0,0)))
        self.objects.append(Plane(origin=(0.0,25,0), normal=(0,1,0), material=Matte(1, numpy.array([1.0,1.0,1.0]))))
        self.objects.append(Sphere(
                    center=(-300, -150, 0),
                    radius=100.0,
                    material=Phong(1,numpy.array([0.8,0.8,0.8]),1)))
        self.objects.append(Sphere(
                    center=(-75, -150, 0),
                    radius=100.0,
                    material=Matte(1,numpy.array([0.8,0.8,0.8]))))
        self.objects.append(Sphere(
                    center=(75, -150, 0),
                    radius=100.0,
                    material=Phong(1,numpy.array([0.8,0.8,0.8]),2)))
        self.objects.append(Sphere(
                    center=(300, -150, 0),
                    radius=100.0,
                    material=Phong(1,numpy.array([0.8,0.8,0.8]),3)))


    def hit_bare_bones_objects(self, ray):
        tmin = INF
        hit = None
        for obj in self.objects:
            shader_rec = obj.hit(ray)
            if shader_rec and shader_rec.tmin < tmin:
                hit = shader_rec
                tmin = shader_rec.tmin
                shader_rec.color = obj.get_color()
        return hit

    def hit_objects(self, ray):

        tmin = INF
        hit = None
        for obj in self.objects:
            shader_rec = obj.hit(ray)
            if shader_rec and shader_rec.tmin < tmin:
                hit = shader_rec
                tmin = shader_rec.tmin
                shader_rec.material = obj.get_material()
                shader_rec.hit_point = ray.origin + tmin * ray.direction
                shader_rec.world = self
        return hit

    def rotate_camera(self, roll, yaw, pitch):
        roll_mat = rotation_matrix(self.camera.w, roll)
        pitch_mat = rotation_matrix(self.camera.u, pitch)
        yaw_mat = rotation_matrix(self.camera.v, yaw)

        rotation = roll_mat.dot(pitch_mat.dot(yaw_mat))
        self.camera.w = rotation.dot(self.camera.w)
        self.camera.u = rotation.dot(self.camera.u)
        self.camera.v = rotation.dot(self.camera.v)

    def move_camera(self, pan):
        self.camera.eye += self.camera.w * -pan

    def render(self):
        pygame.init()

        window = pygame.display.set_mode(self.viewplane.resolution)
        pxarray = pygame.PixelArray(window)
        im = Image.new("RGB", self.viewplane.resolution)

        prev = [-1]
        def render_pixel_offline(pixel,r,g,b):
            im.putpixel(pixel, (r,g,b))
            pxarray[pixel[0]][pixel[1]] = (r,g,b)
            if prev[0] != pixel[1]:
                prev[0] = pixel[1]
                pygame.display.flip()

        def render_pixel_realtime(pixel,r,g,b):
            pxarray[pixel[0]][pixel[1]] = (r,g,b)
            if prev[0] != pixel[1]:
                prev[0] = pixel[1]

        

        need_render = True
        while True:
            for event in pygame.event.get():
                
                if event.type == pygame.KEYDOWN:
                    yaw = 0
                    roll = 0
                    pitch = 0
                    pan = 0
                    if event.key == pygame.K_d:
                        yaw -= 1
                    if event.key == pygame.K_a:
                        yaw += 1

                    if event.key == pygame.K_w:
                        pitch -= 1
                    if event.key == pygame.K_s:
                        pitch += 1

                    if event.key == pygame.K_q:
                        roll -= 1
                    if event.key == pygame.K_e:
                        roll += 1

                    if event.key == pygame.K_j:
                        pan += 50
                    if event.key == pygame.K_k:
                        pan -= 50

                    if event.key == pygame.K_u:
                        self.camera.focal_plane_distance += 100
                        need_render = True
                    if event.key == pygame.K_i:
                        self.camera.focal_plane_distance -= 100
                        need_render = True

                    if yaw != 0 or roll != 0 or pitch != 0 or pan != 0:
                        self.rotate_camera(roll * 0.1, yaw * 0.1, pitch * 0.1)
                        self.move_camera(pan)
                        need_render = True
                if event.type == pygame.QUIT: 
                    sys.exit(0)
            if need_render:
                if self.viewmode == "realtime":
                    self.camera.render(self, render_pixel_realtime)
                else:
                    self.camera.render(self, render_pixel_offline)
                    im.save("render.png", "PNG")
                pygame.display.flip()
                need_render = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--viewmode', dest='viewmode', action='store',
                       default="realtime",
                       help='View mode: realtime or offline')

    args = parser.parse_args()

    w=World(viewmode=args.viewmode)
    w.render()
