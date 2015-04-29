from PIL import Image
import pygame
import math
import numpy
import sys
import random
from sampler import *
from camera import PinholeCamera
from tracer import *

class World(object):
    def __init__(self):
        self.viewplane = ViewPlane(resolution=(320,200), pixel_size=1.0,
            sampler=RegularSampler())
        self.camera = PinholeCamera(eye=(0,0,800), up=(0,1,0), lookat=(0,0,0), viewing_distance=200)
        self.background_color = (0.0,0.0,0.0)
        self.tracer = Tracer(self)
        self.objects = []
        # initiate objects
        for x in xrange(3):
            for y in xrange(3):
                self.objects.append(Sphere(center=(x * 250 - 250,y * 120 - 150,500.0), radius=50.0, color=(x / 3., y / 3., .5)))
        #self.objects.append(Sphere(center=(50.0,10.0,500.0), radius=85.0, color=(1.0,1.0,0)))
        #self.objects.append(Plane(origin=(0.0,25,0), normal=(0,1,0), color=(0,0,1.0)))

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

    def render(self):
        pygame.init()

        window = pygame.display.set_mode(self.viewplane.resolution)
        pxarray = pygame.PixelArray(window)
        im = Image.new("RGB", self.viewplane.resolution)

        prev = [-1]
        def render_pixel(pixel,r,g,b):
            im.putpixel(pixel, (r,g,b))
            pxarray[pixel[0]][pixel[1]] = (r,g,b)
            if prev[0] != pixel[1]:
                prev[0] = pixel[1]
                pygame.display.flip()

        self.camera.render(self, render_pixel)
        im.save("render.png", "PNG")

        while True: 
           for event in pygame.event.get(): 
              if event.type == pygame.QUIT: 
                  sys.exit(0)

if __name__ == "__main__":
    w=World()
    w.render()
