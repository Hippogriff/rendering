# rendering
Quick trimesh headless rendering. Supports depth map, normal map, ray to mesh intersection point and triangle ids.

Usage:

````Python
import trimesh
from matplotlib import pyplot as plt
mesh = trimesh.load("../selfsup-proj/objs/160372_3.obj")
from render import render_mesh
images = render_mesh(mesh)
plt.imshow(images[20])
````
