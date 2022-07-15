import os
import matplotlib
from render_utils import *
from render_utils import create_uniform_camera_poses


def create_uniform_camera_poses_circular(N, r=2):
    mesh = geometry.TriangleMesh()

    def rotation_matrix_y(degrees):
        rad = degrees / 180 * np.pi
        return np.array(
            [[np.cos(rad), 0, -np.sin(rad)], [0, 1, 0], [np.sin(rad), 0, np.cos(rad)]]
        )

    camera_poses = []
    for i in range(N):
        frontvectors = np.array([0, 0, 1]) * r
        frontvectors = rotation_matrix_y(360 / N * i) @ frontvectors
        camera_pose = np.array(
            pyrr.Matrix44.look_at(
                eye=frontvectors, target=np.zeros(3), up=np.array([0.0, 1.0, 0])
            ).T
        )
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
    return np.stack(camera_poses, 0)


def render_mesh(mesh, camera_poses=None, resolution=1024, only_render_images=False):
    from render_utils import create_uniform_camera_poses

    if not isinstance(camera_poses, np.ndarray):
        camera_poses = create_uniform_camera_poses(2.0)
        camera_poses = np.stack([camera_poses[i] for i in [0, 17, 22, 59]], 0)
    render = Render(size=resolution, camera_poses=camera_poses)
    triangle_ids, rendered_images, normal_maps, depth_images, p_images = render.render(
        path=None, clean=False, mesh=mesh, only_render_images=only_render_images
    )
    return rendered_images


def render_mesh_circular(mesh, N=10, resolution=1024, only_render_images=False):
    camera_poses = create_uniform_camera_poses_circular(N)
    render = Render(size=resolution, camera_poses=camera_poses)
    triangle_ids, rendered_images, normal_maps, depth_images, p_images = render.render(
        path=None, clean=False, mesh=mesh, only_render_images=only_render_images
    )
    return rendered_images
