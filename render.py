import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import matplotlib
matplotlib.use("Agg")
from render_utils import *


def render_mesh(mesh, camera_poses=None, resolution=1024):
    from render_utils import create_uniform_camera_poses
    if not isinstance(camera_poses, np.ndarray):
        camera_poses = create_uniform_camera_poses(2.0)
        camera_poses = np.stack([camera_poses[i] for i in [17, 22, 59]], 0)
    render = Render(size=resolution, camera_poses=camera_poses)
    triangle_ids, rendered_images, normal_maps, depth_images, p_images = render.render(path=None,
                                                                                       clean=False,
                                                                                       mesh=mesh,
                                                                                       only_render_images=True)
    return rendered_images