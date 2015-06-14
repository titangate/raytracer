import pygame
import numpy
import sys
from geometry import INF
import buildfunctions
import argparse
import affinetransform as transform
import time
from multiprocessing import Process, Queue


class World(object):
    def __init__(self, viewmode="realtime", buildfunction="a", fast=False, breakon=None):
        fcn = buildfunctions.buildfunctionbase.BuildFunctionBase.get_build_function(buildfunction)
        if fcn:
            fcn(self, viewmode)
        else:
            print 'No Buildfunction found! Make sure a buildfunction file exists under buildfunctions/'
            sys.exit()
        self.fast = fast
        self.breakon = breakon

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
                shader_rec.hit_point = ray.origin + tmin * ray.direction
                shader_rec.world = self
        return hit

    def rotate_camera(self, roll, yaw, pitch):
        roll_mat = transform.rotate(self.camera.w, roll)
        pitch_mat = transform.rotate(self.camera.u, pitch)
        yaw_mat = transform.rotate(self.camera.v, yaw)

        rotation = roll_mat.dot(pitch_mat.dot(yaw_mat))
        self.camera.w = transform.apply(rotation, self.camera.w)
        self.camera.u = transform.apply(rotation, self.camera.u)
        self.camera.v = transform.apply(rotation, self.camera.v)

    def move_camera(self, pan):
        self.camera.eye += self.camera.w * -pan

    def render(self):
        pygame.init()

        window = pygame.display.set_mode(self.viewplane.resolution)
        surface = pygame.Surface(self.viewplane.resolution)

        prev = [-1]
        last_time = [time.time()]

        pixel_queue = Queue()
        job_queue = Queue()

        total_pixel_count = [0]

        def render_parallize():
            def dispatch_job(pixel_queue, job_queue, process_id):
                # def render_pixel_dispatch(pixel, r, g, b):
                #     pixel_queue.put([pixel, r, g, b, process_id])
                while True:
                    pairs = job_queue.get()
                    pixel_color_pairs = self.camera.render_pixels(self, pairs)
                    print '%d processed %d pixels' % (process_id, len(pixel_color_pairs))
                    pixel_queue.put(pixel_color_pairs)

            for pixel_plane_point_pairs_chunk in (
                    self.camera.get_pixel_plane_point_pairs(self, 1000)):
                job_queue.put(pixel_plane_point_pairs_chunk)
                total_pixel_count[0] += len(pixel_plane_point_pairs_chunk)

            for i in xrange(4):
                p = Process(target=dispatch_job, args=(pixel_queue, job_queue, i))
                p.start()

        def render_pixel_offline(pixel, r, g, b):
            r = min(r, 255)
            g = min(g, 255)
            b = min(b, 255)
            surface.set_at(pixel, (r, g, b))
            if numpy.abs(prev[0] - pixel[1]) > 5:
                prev[0] = pixel[1]
                window.blit(surface, (0, 0))
                pygame.display.flip()
                print 'time elapsed: %.2f' % (time.time() - last_time[0])
                last_time[0] = time.time()

        def render_pixel_realtime(pixel, r, g, b):
            r = min(r, 255)
            g = min(g, 255)
            b = min(b, 255)
            surface.set_at(pixel, (r, g, b))

        need_render = True
        running_dispatch = False
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
                need_render = False
                if self.viewmode == "realtime":
                    self.camera.render(self, render_pixel_realtime)
                elif self.viewmode == "offline":
                    if self.fast:
                        self.camera.render(self, render_pixel_offline)
                    else:
                        self.camera.render_progressive(self, render_pixel_offline)
                else:
                    render_parallize()
                    running_dispatch = True
                    continue

                window.blit(surface, (0, 0))
                pygame.display.flip()
                pygame.image.save(surface, "render.png")
                print 'render complete!'

            if running_dispatch:
                print 'waiting for pixels...'
                new_pixels = pixel_queue.get()
                total_pixel_count[0] -= len(new_pixels)
                print 'got pixels'
                for pixel, r, g, b in new_pixels:
                    render_pixel_realtime(pixel, r, g, b)
                window.blit(surface, (0, 0))
                pygame.display.flip()
                if total_pixel_count[0] == 0:
                    running_dispatch = False
                    window.blit(surface, (0, 0))
                    pygame.display.flip()
                    pygame.image.save(surface, "render.png")
                    print 'render complete!'
                    print 'time elapsed: %.2f' % (time.time() - last_time[0])
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--viewmode', dest='viewmode', action='store',
                        default="realtime",
                        help='View mode: realtime or offline')
    parser.add_argument('--buildfunction', dest='buildfunction', action='store',
                        default="a",
                        help='Build Function: a or b')
    parser.add_argument('--fast', dest="fast", action='store_true',
                        default=False)
    parser.add_argument('--breakon', dest='breakon', action='store',
                        default=None,
                        help='break on a pixel. e.g: 100,200')

    args = parser.parse_args()

    breakon = None

    if args.breakon:
        breakon = map(int,args.breakon.split(','))

    w = World(viewmode=args.viewmode, buildfunction=args.buildfunction, fast=args.fast,
              breakon=breakon)
    w.render()
