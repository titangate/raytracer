# raytracer

```
usage: main.py [-h] [--viewmode VIEWMODE] [--numcores NUMCORES]
               [--buildfunction BUILDFUNCTION] [--fast] [--breakon BREAKON]

Render some images.

optional arguments:
  -h, --help            show this help message and exit
  --viewmode VIEWMODE   View mode: realtime or offline or parallel
  --numcores NUMCORES   Number of Cores: number of CPU cores to use(spawn #
                        processes)
  --buildfunction BUILDFUNCTION
                        Build Function: file name of *.py under
                        buildfunctions/
  --fast
  --breakon BREAKON     break on a pixel. e.g: 100,200
```

Example: ``` python main.py --buildfunction=glossyreflective --viewmode=parallel --numcores=2 ```

###View Mode
- realtime: small image, can use wsadqejk to navigate
- offline: bigger image, a few lines at a time scan
- *parallel*(recommended): distribute tasks to parallelized threads. cannot breakpoint.

###Number of cores
just that. has to be integers. should choose it to be equal or less than the number of CPU cores on your computer.

###Build function
each file under ```buildfunctions/``` contains a demo scene. Specify which scene you want to render.

###Fast
deprecated

###Break on
use in realtime/offline mode. setup breakpoint at a certain pixel.

##Feature list
- Sphere, plane, box geometry
- Triangular mesh, kdtree accelerated
- Matte, Phong material
- Depth of field
- Anti aliasing
- Point light, directional light, ambient light, area light
- Shadow, soft shadow
- Affine transforms, instancing
- Ambient Occulusion, environment light
- Mirror reflection
- Glossy reflection

##Planned
- Global illumination
- Transparency
- Texture
- Distributed worker over clusters

Some demo images:

![](https://github.com/titangate/raytracer/blob/master/demo_images/arealightball.png?raw=true)

![](https://github.com/titangate/raytracer/blob/master/demo_images/dofballs.png?raw=true)

![](https://github.com/titangate/raytracer/blob/master/demo_images/dragon.jpg?raw=true)

![](https://github.com/titangate/raytracer/blob/master/demo_images/glossy.png?raw=true)
