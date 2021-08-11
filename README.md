# rendering
Quick trimesh headless rendering. Supports depth map, normal map, ray to mesh intersection point and triangle ids.

Usage:

````Python
import trimesh
from matplotlib import pyplot as plt
from render import render_mesh

mesh = trimesh.load("path-to-file.obj")
images = render_mesh(mesh)
plt.imshow(images[20])
````
