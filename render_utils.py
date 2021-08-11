import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pyglet

pyglet.options['shadow_window'] = False
import matplotlib

matplotlib.use("Agg")
import open3d as o3d
import torch
from open3d import *
from matplotlib import pyplot as plt
import pyrr
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
)
from sklearn.neighbors import KDTree
import trimesh
import pyrender
import numpy as np
from PIL import Image

import time

SIZE = None
Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
draw_geometries = o3d.visualization.draw_geometries


class Render:
    def __init__(self, size, camera_poses):
        self.size = size
        global SIZE
        SIZE = size

        if not isinstance(camera_poses, np.ndarray):
            self.camera_poses = create_uniform_camera_poses(2.0)
        else:
            self.camera_poses = camera_poses

    def render(self, path, clean=True, intensity=6.0, mesh=None, only_render_images=False):
        # TODO
        # segmentation label transfer
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = prepare_mesh(path, color=False, clean=clean)
        try:
            if mesh.visual.defined:
                mesh.visual.material.kwargs["Ns"] = 1.0
        except:
            print ("Error loading material!")
        mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        t1 = time.time()
        triangle_ids, normal_maps, depth_images, p_images = None, None, None, None
        if not only_render_images:
            # TODO Normals are not normalized.
            triangle_ids, normal_maps, depth_images, p_images = correct_normals(mesh, self.camera_poses, correct=True)
        rendered_images, _ = pyrender_rendering(
                mesh1, viz=False, light=True, camera_poses=self.camera_poses, intensity=intensity
            )
        print(time.time() - t1)
        return triangle_ids, rendered_images, normal_maps, depth_images, p_images


def correct_normals(mesh, camera_poses, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    triangle_images = []
    normalmaps = []
    depth_maps = []
    p_images = []
    for i in range(camera_poses.shape[0]):
        a, b, index_tri, sign, p_image = trimesh_ray_tracing(
            mesh, camera_poses[i], resolution=SIZE, rayintersector=rayintersector
        )
        if correct:
            mesh.faces[index_tri[sign > 0]] = np.fliplr(mesh.faces[index_tri[sign > 0]])

        normalmap = render_normal_map(
            pyrender.Mesh.from_trimesh(mesh, smooth=False),
            camera_poses[i],
            SIZE,
            viz=False,
        )

        triangle_images.append(b)
        normalmaps.append(normalmap)
        depth_maps.append(a)
        p_images.append(p_image)
    return triangle_images, normalmaps, depth_maps, p_images


def all_rendering(mesh, camera_poses, light=False, viz=False, correct=True):
    rayintersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    mesh1 = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene()
    scene.add(mesh1)
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    # light
    if light:
        lights = init_light(scene, camera_poses[0])

    triangle_images = []
    normalmaps = []
    depth_maps = []
    color_images = []

    for i in range(camera_poses.shape[0]):
        a, b, index_tri, sign = trimesh_ray_tracing(
            mesh, camera_poses[i], resolution=SIZE, rayintersector=rayintersector
        )
        if correct:
            mesh.faces[index_tri[sign > 0]] = np.fliplr(mesh.faces[index_tri[sign > 0]])

        normalmap = render_normal_map(
            pyrender.Mesh.from_trimesh(mesh, smooth=False),
            camera_poses[i],
            SIZE,
            viz=False,
        )

        if light:
            update_light(scene, lights, camera_poses[i])

        if light:
            color, _ = r.render(scene
                                )  # , flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES
        else:
            color, _ = r.render(
                scene, flags=pyrender.constants.RenderFlags.FLAT
            )  # | pyrender.constants.RenderFlags.SKIP_CULL_FACES

        triangle_images.append(b)
        normalmaps.append(normalmap)
        depth_maps.append(a)
        color_images.append(color)
    return color_images, triangle_images, normalmaps, depth_maps


def normalize_mesh(mesh, mode="sphere"):
    if mode == "sphere":
        mesh.vertices = mesh.vertices - mesh.vertices.mean(0)
        scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale
    elif mode == "com":
        box = mesh.bounding_box_oriented
        mesh.vertices = mesh.vertices - box.vertices.mean(0)
        scale = np.linalg.norm(mesh.vertices, axis=1, ord=2).max()
        mesh.vertices = mesh.vertices / scale


def prepare_mesh(model_name, color=False, clean=False):
    mesh = trimesh.load(model_name, force="mesh")
    # mesh = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.faces))
    # if remesh:
    #     v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, 0.1)
    #     mesh.vertices = v
    #     mesh.faces = f
    if clean:
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()

        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_normals(mesh)

    normalize_mesh(mesh, "com")
    if color:
        mesh.visual.face_colors = generate_unique_colors(
            mesh.faces.shape[0]
        )
    return mesh


def clean_using_o3d(mesh):
    mesh = convert_trimesh_to_o3d(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    p = mesh.sample_points_poisson_disk(10000, 1)
    o3d.visualization.draw_geometries([mesh, p])
    return convert_o3d_to_trimesh(mesh)


def generate_unique_colors(size):
    colors = np.arange(1, 254 * 254 * 254)
    z = np.random.choice(colors, (size), replace=False)
    colors = np.unravel_index(z, (255, 255, 255))
    colors = np.stack(colors, 1)
    return colors


def draw_lines(b1, b2):
    b1 = b1.reshape((SIZE * SIZE, 1)).astype(np.int32)
    b2 = b2.reshape((SIZE * SIZE, 1)).astype(np.int32)
    tree = KDTree(b1)
    d, indices = tree.query(b2)
    return d, indices


def init_light(scene, camera_pose, intensity=6.0):
    direc_l = DirectionalLight(color=np.ones(3), intensity=intensity)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    point_l = PointLight(color=np.ones(3), intensity=1)

    direc_l_node = scene.add(direc_l, pose=camera_pose)
    point_l_node = scene.add(point_l, pose=camera_pose)
    spot_l_node = scene.add(spot_l, pose=camera_pose)
    return spot_l_node, direc_l_node, point_l_node


def update_light(scene, lights, pose):
    for l in lights:
        scene.set_pose(l, pose)


class CustomShaderCache:
    def __init__(self):
        self.program = None

    def get_program(
            self, vertex_shader, fragment_shader, geometry_shader=None, defines=None
    ):
        if self.program is None:
            self.program = pyrender.shader_program.ShaderProgram(
                "shades/mesh.vert", "shades/mesh.frag", defines=defines
            )
        return self.program


def render_normal_map(mesh, camera_pose, size, viz=False):
    scene = pyrender.Scene(bg_color=(255, 255, 255))
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(camera, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(size, size)
    renderer._renderer._program_cache = CustomShaderCache()

    normals, depth = renderer.render(
        scene
    )  # flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES
    world_space_normals = normals / 255 * 2 - 1

    if viz:
        image = Image.fromarray(normals, "RGB")
        image.show()

    return world_space_normals


def pyrender_rendering(mesh, camera_poses, viz=False, light=False, intensity=6.0):
    # renderer
    r = pyrender.OffscreenRenderer(SIZE, SIZE)

    scene = pyrender.Scene()
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.)
    # frontVector =  np.array( [1.2, 1.2, 1.2] )
    # frontVector = (Rotation.from_euler('y', 0, degrees=True)).apply( frontVector )
    # camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontVector, target=np.zeros(3), up=np.array([0.0, 1.0, 0])).T)
    # camera_pose = np.linalg.inv(np.array(camera_pose))
    # camera_poses = []
    camera = scene.add(camera, pose=camera_poses[0])
    # light
    if light:
        lights = init_light(scene, camera_poses[0], intensity=intensity)

    images = []
    depth_images = []
    for i in range(camera_poses.shape[0]):
        # camera
        # frontVector =  np.array( [1.2, 1.2, 1.2] )
        # frontVector = (Rotation.from_euler('y', 30 * i, degrees=True)).apply( frontVector )
        # camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontVector, target=np.zeros(3), up=np.array([0.0, 1.0, 0])).T)
        # camera_pose = np.linalg.inv(np.array(camera_pose))
        scene.set_pose(camera, camera_poses[i])
        if light:
            update_light(scene, lights, camera_poses[i])

        if light:
            color, depth = r.render(scene
                                    )  # , flags=pyrender.constants.RenderFlags.SKIP_CULL_FACES
        else:
            color, depth = r.render(
                scene, flags=pyrender.constants.RenderFlags.FLAT
            )  # | pyrender.constants.RenderFlags.SKIP_CULL_FACES

        if viz:
            plt.figure()
            plt.imshow(color)
        images.append(color)
        depth_images.append(depth)
    return images, depth_images


def create_look_at(eye, target, up, dtype=None):
    """Creates a look at matrix according to OpenGL standards.

    :param numpy.array eye: Position of the camera in world coordinates.
    :param numpy.array target: The position in world coordinates that the
        camera is looking at.
    :param numpy.array up: The up vector of the camera.
    :rtype: numpy.array
    :return: A look at matrix that can be used as a viewMatrix
    """

    def normalize(a):
        return a / (np.linalg.norm(a, ord=2) + 1e-7)

    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)

    cameraDirection = normalize(eye - target)
    right = normalize(np.cross(normalize(up), cameraDirection))
    up = normalize(np.cross(cameraDirection, right))
    M = np.zeros((4, 4))
    M[0, 0:3] = right
    M[1, 0:3] = up
    M[2, 0:3] = cameraDirection
    M[3, 3] = 1.0
    T = np.eye(4)
    T[0:3, -1] = -eye
    return M @ T


def camera_transform_matrix(eye, target, up, dtype=None):
    """Creates a look at matrix according to OpenGL standards.

    :param numpy.array eye: Position of the camera in world coordinates.
    :param numpy.array target: The position in world coordinates that the
        camera is looking at.
    :param numpy.array up: The up vector of the camera.
    :rtype: numpy.array
    :return: A look at matrix that can be used as a viewMatrix
    """

    def normalize(a):
        return a / (np.linalg.norm(a, ord=2) + 1e-7)

    eye = np.asarray(eye)
    target = np.asarray(target)
    up = np.asarray(up)

    cameraDirection = normalize(eye - target)
    right = -normalize(np.cross(normalize(up), cameraDirection))
    up = -normalize(np.cross(cameraDirection, right))
    M = np.zeros((4, 4))
    M[0, 0:3] = right
    M[1, 0:3] = up
    M[2, 0:3] = cameraDirection
    M[3, 3] = 1.0

    M = M.T
    M = M
    #     M = np.eye(4)
    M[0:3, -1] = eye
    return M


def trimesh_ray_tracing(mesh, M, resolution=225, fov=60, rayintersector=None):
    # this is done to correct the mistake in way trimesh raycasting works.
    # in general this cannot be done.
    extra = np.eye(4)
    extra[0, 0] = 0
    extra[0, 1] = 1
    extra[1, 0] = -1
    extra[1, 1] = 0
    scene = mesh.scene()

    # np.linalg.inv(create_look_at(frontVector, np.zeros(3), np.array([0, 1, 0])))
    scene.camera_transform = M @ extra  # @ np.diag([1, -1,-1, 1]
    # scene.camera_transform = camera_transform_matrix(frontVector, np.zeros(3), np.array([0, 1, 0])) @ e

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    scene.camera.resolution = [resolution, resolution]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    scene.camera.fov = fov, fov

    # convert the camera to rays with one ray per pixel
    origins, vectors, pixels = scene.camera_rays()

    # do the actual ray- mesh queries
    #     points, index_ray, index_tri = mesh.ray.intersects_location(
    #         origins, vectors, multiple_hits=False)
    #     points, index_ray, index_tri = rayintersector.intersects_location(
    #         origins, vectors, multiple_hits=False)

    # for each hit, find the distance along its vector
    index_tri, index_ray, points = rayintersector.intersects_id(
        origins, vectors, multiple_hits=False, return_locations=True
    )
    depth = trimesh.util.diagonal_dot(points - origins[0], vectors[index_ray])
    sign = trimesh.util.diagonal_dot(mesh.face_normals[index_tri], vectors[index_ray])

    # find pixel locations of actual hits
    pixel_ray = pixels[index_ray]
    # create a numpy array we can turn into an image
    # doing it with uint8 creates an `L` mode greyscale image
    a = np.zeros(scene.camera.resolution, dtype=np.uint8)
    b = np.ones(scene.camera.resolution, dtype=np.int32) * -1
    p_image = np.ones([scene.camera.resolution[0], scene.camera.resolution[1], 3], dtype=np.float32) * -1
    # scale depth against range (0.0 - 1.0)
    # import ipdb; ipdb.set_trace()
    depth_float = (depth - depth.min()) / depth.ptp()

    # convert depth into 0 - 255 uint8
    depth_int = (depth_float * 255).round().astype(np.uint8)

    # assign depth to correct pixel locations
    a[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
    b[pixel_ray[:, 0], pixel_ray[:, 1]] = index_tri
    p_image[pixel_ray[:, 0], pixel_ray[:, 1]] = points

    # show the resulting image
    return a, b, index_tri, sign, p_image


def create_uniform_camera_poses(distance=2):
    mesh = geometry.TriangleMesh()
    frontvectors = np.array(mesh.create_sphere(distance, 7).vertices)
    camera_poses = []
    for i in range(frontvectors.shape[0]):
        camera_pose = np.array(pyrr.Matrix44.look_at(eye=frontvectors[i],
                                                     target=np.zeros(3),
                                                     up=np.array([0.0, 1.0, 0])).T)
        camera_pose = np.linalg.inv(np.array(camera_pose))
        camera_poses.append(camera_pose)
    return np.stack(camera_poses, 0)


def generate_dodecahedron():
    # r = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        -0.57735, -0.57735, 0.57735,
        0.934172, 0.356822, 0,
        0.934172, -0.356822, 0,
        -0.934172, 0.356822, 0,
        -0.934172, -0.356822, 0,
        0, 0.934172, 0.356822,
        0, 0.934172, -0.356822,
        0.356822, 0, -0.934172,
        -0.356822, 0, -0.934172,
        0, -0.934172, -0.356822,
        0, -0.934172, 0.356822,
        0.356822, 0, 0.934172,
        -0.356822, 0, 0.934172,
        0.57735, 0.57735, -0.57735,
        0.57735, 0.57735, 0.57735,
        -0.57735, 0.57735, -0.57735,
        -0.57735, 0.57735, 0.57735,
        0.57735, -0.57735, -0.57735,
        0.57735, -0.57735, 0.57735,
        -0.57735, -0.57735, -0.57735,
    ]).reshape((-1, 3), order="C")
    return vertices


def transfer_labels_shapenet_points_to_mesh(points, labels, mesh):
    pcd = visualize_point_cloud(points, viz=False)
    box = pcd.get_axis_aligned_bounding_box()
    points = points @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
    points = points / np.linalg.norm(points, axis=1, ord=2).max()
    points = points - points.mean(0)
    points = points + np.array(mesh.sample(2500)).mean(0)
    pcd.points = Vector3dVector(points)

    _, indices = find_match(np.array(pcd.points), mesh.triangles_center)
    return labels[indices]


def find_match(source, target, k=1):
    tree = KDTree(source)
    d, indices = tree.query(target, k=k)
    return d[:, 0], indices[:, 0]


def normalize_colors(c):
    c = c - c.min((0, 1), keepdims=True)
    c = c / c.max((0, 1), keepdims=True)
    return c


def find_correspondence_bw_images(triangle_ids1, triangle_ids2, return_outside=False):
    if len(triangle_ids1.shape) == 2:
        triangle_ids1 = np.expand_dims(triangle_ids1, 2)
        triangle_ids2 = np.expand_dims(triangle_ids2, 2)

    x_1, y_1 = np.where(triangle_ids1[:, :, 0] > -1)
    triangle_index_1 = triangle_ids1[x_1, y_1]
    x_2, y_2 = np.where(triangle_ids2[:, :, 0] > -1)
    triangle_index_2 = triangle_ids2[x_2, y_2]
    d, indices = find_match(triangle_index_1, triangle_index_2)
    matched_indices_2 = np.where(d < 5e-3)[0]

    matched_indices_1 = indices[matched_indices_2]

    matched_x_2 = x_2[matched_indices_2]
    matched_y_2 = y_2[matched_indices_2]

    matched_x_1 = x_1[matched_indices_1]
    matched_y_1 = y_1[matched_indices_1]

    if return_outside:
        outside_indices_2 = np.where(d > 8e-3)[0]
        return matched_x_1, matched_y_1, matched_x_2, matched_y_2, triangle_ids1[matched_x_1, matched_y_1][:, 0], \
               triangle_ids2[matched_x_2, matched_y_2][:, 0], x_2[outside_indices_2], y_2[outside_indices_2]
    # assert np.sum(triangle_ids1[matched_x_1, matched_y_1] == triangle_ids2[matched_x_2, matched_y_2]) == \
    #        matched_indices_2.shape[0]
    return matched_x_1, matched_y_1, matched_x_2, matched_y_2, triangle_ids1[matched_x_1, matched_y_1][:, 0], triangle_ids2[
        matched_x_2, matched_y_2][:, 0]


def draw_markers1(image1, x1, y1, image2, x2, y2):
    img1 = np.copy(image1)
    img2 = np.copy(image2)

    img1 = np.stack([img1] * 3, 2) / img1.max()
    img2 = np.stack([img2] * 3, 2) / img2.max()
    print(x1, y1, x2, y2, img1[x1, y1], img2[x2, y2])

    img1[x1 - 2:x1 + 2, y1 - 2:y1 + 2] = np.array([250, 0, 0])
    img2[x2 - 2:x2 + 2, y2 - 2:y2 + 2] = np.array([250, 0, 0])
    plt.figure()
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)


def draw_markers(image1, x1, y1, image2, x2, y2):
    img1 = np.copy(image1)
    img2 = np.copy(image2)

    #     img1 = np.stack([img1] * 3, 2)
    #     img2 = np.stack([img2] * 3, 2)
    img1[x1 - 2:x1 + 2, y1 - 2:y1 + 2] = np.array([250, 0, 0])
    img2[x2 - 2:x2 + 2, y2 - 2:y2 + 2] = np.array([250, 0, 0])
    plt.figure()
    plt.imshow(img1)
    plt.figure()
    plt.imshow(img2)


def draw_all_lines(img1, img2, x1, y1, x2, y2, N=1):
    """
    Draws all correspondences in (x1,y1) and (x2,y2).
    Note these coordinates are inverted to be used in
    plotting and not for indexing the array. Replace
    x->y for array indexing.
    """
    img = np.concatenate([img1, img2], 1)
    indices = np.random.choice(x1.shape[0], N)
    for i in indices:
        x = [x1[i], x2[i] + 256]
        y = [y1[i], y2[i]]
        c = np.random.rand(3)
        plt.plot(x, y, color="r", linewidth=1)
        plt.plot(x, y, '.', color="r", linewidth=1)
    plt.imshow(img)
    return img


def draw_prediction(img1, img2, x1, y1, x2, y2):
    """
    Draws all correspondences in (x1,y1) and (x2,y2).
    Note these coordinates are inverted to be used in
    plotting and not for indexing the array. Replace
    x->y for array indexing.
    """
    img = np.concatenate([img1, img2], 1)

    x = [x1[0], x2[0] + 256]
    y = [y1[0], y2[0]]
    c = np.random.rand(3)
    plt.plot(x, y, color="k", linewidth=1)
    plt.plot(x, y, 'bo', color="k", linewidth=1)

    x = [x1[1], x2[1] + 256]
    y = [y1[1], y2[1]]
    c = np.random.rand(3)
    plt.plot(x, y, color="g", linewidth=1)
    plt.plot(x, y, 'bo', color="g", linewidth=1)

    plt.imshow(img)


def draw_random_lines(img1, img2, x1, y1, x2, y2):
    """
    Draws random correspondences in (x1,y1) and (x2,y2).
    Note these coordinates are inverted to be used in
    plotting and not for indexing the array. Replace
    x->y for array indexing.
    """
    img = np.concatenate([img1, img2], 1)
    i = np.random.choice(x1.shape[0], 1)
    x = [x1[i], x2[i] + 256]
    y = [y1[i], y2[i]]
    c = np.random.rand(3)
    plt.plot(x, y, color=c, linewidth=1)
    plt.plot(x, y, 'o', color=c, linewidth=1)
    plt.imshow(img)


def find_correspondence_in_feature_space(f1, f2, x1, y1, x2, y2):
    f1 = torch.nn.functional.normalize(f1, dim=1, p=2).permute((1, 2, 0))
    f2 = torch.nn.functional.normalize(f2, dim=1, p=2).permute((1, 2, 0))
    f1 = f1[x1, y1]
    f2 = f2[x2, y2]

    dist = f1 @ f2.T
    indices = torch.max(dist, axis=1)[1]
    return indices, dist


def heatmap(image, x, y, weights):
    cmap = plt.cm.get_cmap("seismic", 256)
    img = image.copy()
    weights = weights.reshape((-1, 1)) - weights.min()
    weights = weights ** 2
    weights = weights / weights.max()
    weights = weights * 255
    weights = weights.astype(np.int32)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]

    color = cmap[weights]
    color = np.squeeze(color, 1)
    # img[x, y] = np.array([[0, 0, 1]]) * weights
    img[x, y] = color
    plt.imshow(img)


def find_correspondence_in_feature_space_eval_metric(f1, f2, tri1, tri2):
    """
    At higher level we want to find the closest point in image 2
    w.r.t a point in image 1. For the first image we select pixels
    that have at least one matching pixel in another image. Starting
    with these pixels, we find closest pixel in second image. However,
    we only take points from the foreground for second image.
    Parameters
    ----------
    f1
    f2
    tri1
    tri2

    Returns
    -------
    x2_f: pixels coordinate x in the second image that matches with pixel coordinate
    x_1 in the first image based on feature similarity.
    x2: pixels coordinate x in the second image that matches with pixel coordinate
    x_1 in the first image based ground truth data.
    """
    x1, y1, x2, y2, _, _ = find_correspondence_bw_images(tri1, tri2)
    x1 = x1
    y1 = y1

    x2_org, y2_org = np.where(tri2[:, :, 0] > -1)

    f1 = f1.permute((1, 2, 0))
    f2 = f2.permute((1, 2, 0))

    f1 = f1[x1, y1]
    f2 = f2[x2_org, y2_org]

    f1 = torch.nn.functional.normalize(f1, dim=1, p=2)
    f2 = torch.nn.functional.normalize(f2, dim=1, p=2)
    dist = f1 @ f2.T
    indices = torch.max(dist, axis=1)[1]
    x2_f = x2_org[indices.data.cpu().numpy()]
    y2_f = y2_org[indices.data.cpu().numpy()]
    return x1, y1, x2, y2, x2_f, y2_f, x2_org, y2_org, dist


def find_correspondence_in_feature_space_all_pixels(f1, f2, tri1, tri2):
    """
    At higher level we want to find the closest point in image 2
    w.r.t a point in image 1. For the first image we select pixels
    that have at least one matching pixel in another image. Starting
    with these pixels, we find closest pixel in second image. However,
    we only take points from the foreground for second image.
    Parameters
    ----------
    f1
    f2
    tri1
    tri2

    Returns
    -------
    x2_f: pixels coordinate x in the second image that matches with pixel coordinate
    x_1 in the first image based on feature similarity.
    x2: pixels coordinate x in the second image that matches with pixel coordinate
    x_1 in the first image based ground truth data.
    """
    x1_org, y1_org = np.where(tri1 > -1)
    x2_org, y2_org = np.where(tri2 > -1)

    f1 = f1.permute((1, 2, 0))
    f2 = f2.permute((1, 2, 0))

    f1 = f1[x1_org, y1_org]
    f2 = f2[x2_org, y2_org]

    f1 = torch.nn.functional.normalize(f1, dim=1, p=2)
    f2 = torch.nn.functional.normalize(f2, dim=1, p=2)
    dist = f1 @ f2.T
    indices = torch.max(dist, axis=1)[1]
    x2_f = x2_org[indices.data.cpu().numpy()]
    y2_f = y2_org[indices.data.cpu().numpy()]

    x1_corr, y1_corr, x2_corr, y2_corr, _, _ = find_correspondence_bw_images(tri1, tri2)
    return x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr


def find_correspondence_in_feature_space_point_image(f1, f2, tri1, tri2):
    """
    At higher level we want to find the closest point in image 2
    w.r.t a point in image 1. For the first image we select pixels
    that have at least one matching pixel in another image. Starting
    with these pixels, we find closest pixel in second image. However,
    we only take points from the foreground for second image.
    Parameters
    ----------
    f1
    f2
    tri1
    tri2

    Returns
    -------
    x2_f: pixels coordinate x in the second image that matches with pixel coordinate
    x_1 in the first image based on feature similarity.
    x2: pixels coordinate x in the second image that matches with pixel coordinate
    x_1 in the first image based ground truth data.
    """
    x1_org, y1_org = np.where(tri1[:, :, 0] > -1)
    x2_org, y2_org = np.where(tri2[:, :, 0] > -1)

    f1 = f1.permute((1, 2, 0))
    f2 = f2.permute((1, 2, 0))

    f1 = f1[x1_org, y1_org]
    f2 = f2[x2_org, y2_org]

    f1 = torch.nn.functional.normalize(f1, dim=1, p=2)
    f2 = torch.nn.functional.normalize(f2, dim=1, p=2)
    dist = f1 @ f2.T
    dist, indices = torch.max(dist, axis=1)
    x2_f = x2_org[indices.data.cpu().numpy()]
    y2_f = y2_org[indices.data.cpu().numpy()]

    x1_corr, y1_corr, x2_corr, y2_corr, _, _ = find_correspondence_bw_images(tri1, tri2)
    return x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr, dist


def return_closes_points(out1, out2, x1, y1, x2, y2):
    f1 = out1.permute((1, 2, 0))
    f2 = out2.permute((1, 2, 0))
    f1 = f1[x1, y1]
    f2 = f2[x2, y2]

    f1 = torch.nn.functional.normalize(f1, dim=1, p=2)
    f2 = torch.nn.functional.normalize(f2, dim=1, p=2)

    dist = f1 @ f2.T
    dist, indices = torch.max(dist, axis=1)
    return dist, indices

def visualize_dense_correspondence(out1, out2, triangle_1, triangle_2, img1, img2, face_coords):
    """
    Here we want to visualize dense correspondence. We start by coloring the mesh face centers
    with coordinate based colors. Then for the first view, we take all foreground pixels and
    find their correspondence with the second view using similarity in feature space. Then we
    color the corresponding pixels in the second view by borrowing the colors from the first
    view.
    Parameters
    ----------
    out1
    out2
    triangle_1
    triangle_2
    img1
    img2
    face_coords

    Returns
    -------

    """
    images = []
    x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr = find_correspondence_in_feature_space_all_pixels(out1,
                                                                                                             out2,
                                                                                                             triangle_1,
                                                                                                             triangle_2)

    # First view, coordinate color maps for all foreground pixels
    fig, ax = plt.subplots(1, 3, figsize=(6, 1), dpi=200)
    # fig.tight_layout()
    colors1 = face_coords[triangle_1[x1_org, y1_org]]
    temp_image = img1.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x1_org, y1_org] = colors1
    images.append(temp_image)
    ax[0].imshow(temp_image)

    # Second image, visualize the coordinate colors for all foreground pixels
    colors1 = face_coords[triangle_2[x2_org, y2_org]]
    temp_image = img2.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x2_org, y2_org] = colors1
    images.append(temp_image)
    ax[1].imshow(temp_image)

    # second image, correspondence maps for all forground pixels in the first image
    colors1 = face_coords[triangle_1[x1_org, y1_org]]
    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image = img2.copy()
    temp_image[x2_f, y2_f] = colors1
    images.append(temp_image)
    ax[2].imshow(temp_image)


    # First image, coordinate color map for pixels that are in correspondence between two images
    # colors1 = face_coords[triangle_1[x1_corr, y1_corr]]
    # temp_image = img1.copy()
    #
    # colors1 = colors1 - colors1.min(0)
    # colors1 = colors1 / colors1.max(0)
    #
    # temp_image[x1_corr, y1_corr] = colors1
    # ax[3].imshow(temp_image)

    # Second image, ground truth coordinate color maps for pixels that are in correspondence between two images
    # temp_image = img2.copy()
    # temp_image[x2_corr, y2_corr] = colors1
    # ax[4].imshow(temp_image)

    for a in ax:
        a.axis("off")

    # plt.savefig("results/heatmap_all.png")
    return images


def visualize_dense_correspondence_point_image_cross_shape(out1, out2, triangle_1, triangle_2, point_image1, point_image2, img1, img2, path=None, viz=False):
    """
    Here we want to visualize dense correspondence. We start by coloring the mesh face centers
    with coordinate based colors. Then for the first view, we take all foreground pixels and
    find their correspondence with the second view using similarity in feature space. Then we
    color the corresponding pixels in the second view by borrowing the colors from the first
    view.
    Parameters
    ----------
    out1
    out2
    triangle_1
    triangle_2
    img1
    img2
    face_coords

    Returns
    -------

    """
    mesh = prepare_mesh(path)
    face_coords = mesh.triangles_center

    images = []
    # x1_org, y1_org = np.where(triangle_1 > -1)
    x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr = find_correspondence_in_feature_space_point_image(
        out1,
        out2,
        point_image1,
        point_image2)
    # First view, coordinate color maps for all foreground pixels
    if viz:
        fig, ax = plt.subplots(1, 2, figsize=(6, 1), dpi=200)
    # fig.tight_layout()
    colors1 = face_coords[triangle_1[x1_org, y1_org]]
    temp_image = img1.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x1_org, y1_org] = colors1
    images.append(temp_image)
    if viz:
        ax[0].imshow(temp_image)

    temp_image = img2.copy()
    temp_image[x2_f, y2_f] = colors1
    images.append(temp_image)
    if viz:
        ax[1].imshow(temp_image)

    if viz:
        for a in ax:
            a.axis("off")
    return images


def prune_outliers_dense_correspondence(out1, out2, triangle_1, triangle_2, point_image1, point_image2, img1, img2, viz=False, face_coords=None, path=None):
    x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr = find_correspondence_in_feature_space_point_image(out1,
                                                                                                             out2,
                                                                                                             point_image1,
                                                                                                             point_image2)

    dist, indices = return_closes_points(out2, out1, x2_f, y2_f, x1_org, y1_org)
    dist = dist.data.cpu().numpy()
    indices = indices.data.cpu().numpy()

    x1_cycle, y1_cycle = x1_org[indices], y1_org[indices]
    x_cycle = np.stack([x1_cycle, y1_cycle], 1)
    x = np.stack([x1_org, y1_org], 1)

    indices = np.where(dist > 0.8)[0]

    images = []
    # First view, coordinate color maps for all foreground pixels
    if viz:
        fig, ax = plt.subplots(1, 3, figsize=(6, 1), dpi=200)
    # fig.tight_layout()
    colors1 = face_coords[triangle_1[x1_org, y1_org]]
    temp_image = img1.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x1_org, y1_org] = colors1
    images.append(temp_image)
    if viz:
        ax[0].imshow(temp_image)

    # Second image, visualize the coordinate colors for all foreground pixels
    colors1 = face_coords[triangle_2[x2_org, y2_org]]
    temp_image = img2.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x2_org, y2_org] = colors1
    images.append(temp_image)
    if viz:
        ax[1].imshow(temp_image)

    # second image, correspondence maps for all forground pixels in the first image
    colors1 = face_coords[triangle_1[x1_org[indices], y1_org[indices]]]
    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image = img2.copy()
    temp_image[x2_f[indices], y2_f[indices]] = colors1
    images.append(temp_image)
    if viz:
        ax[2].imshow(temp_image)
    return images


def visualize_dense_correspondence_point_image_backward(out1, out2, triangle_1, triangle_2, point_image1, point_image2, img1, img2, face_coords, viz=False):
    """
    Here we want to visualize dense correspondence. We start by coloring the mesh face centers
    with coordinate based colors. Then for the first view, we take all foreground pixels and
    find their correspondence with the second view using similarity in feature space. Then we
    color the corresponding pixels in the second view by borrowing the colors from the first
    view.
    Parameters
    ----------
    out1
    out2
    triangle_1
    triangle_2
    img1
    img2
    face_coords

    Returns
    -------

    """
    images = []
    x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr, dist = \
        find_correspondence_in_feature_space_point_image(out1,
                                                         out2,
                                                         point_image1,
                                                         point_image2)
    dist = dist.data.cpu().numpy()
    valid_indices = np.where(dist > 0.7)[0]
    # valid_indices = np.arange(dist.shape[0])
    # First view, coordinate color maps for all foreground pixels
    if viz:
        fig, ax = plt.subplots(1, 3, figsize=(6, 1), dpi=200)
    # fig.tight_layout()
    colors1 = face_coords[triangle_1[x1_org, y1_org]]
    temp_image = img1.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x1_org, y1_org] = colors1
    images.append(temp_image)
    if viz:
        ax[0].imshow(temp_image)

    # Second image, visualize the coordinate colors for all foreground pixels
    colors1 = face_coords[triangle_2[x2_org, y2_org]]
    temp_image = img2.copy()

    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image[x2_org, y2_org] = colors1
    images.append(temp_image)
    if viz:
        ax[1].imshow(temp_image)

    # second image, correspondence maps for all forground pixels in the first image
    colors1 = face_coords[triangle_2[x2_f[valid_indices], y2_f[valid_indices]]]
    colors1 = colors1 - colors1.min(0)
    colors1 = colors1 / colors1.max(0)

    temp_image = img1.copy()
    temp_image[x1_org[valid_indices], y1_org[valid_indices]] = colors1
    images.append(temp_image)

    if viz:
        ax[2].imshow(temp_image)
    if viz:
        for a in ax:
            a.axis("off")
    return images


def visualize_dense_correspondence_point_image_backward_texture(out1, out2, triangle_1, triangle_2, point_image1, point_image2, img1, img2, face_coords, viz=False):
    """
    Here we want to visualize dense correspondence. We start by coloring the mesh face centers
    with coordinate based colors. Then for the first view, we take all foreground pixels and
    find their correspondence with the second view using similarity in feature space. Then we
    color the corresponding pixels in the second view by borrowing the colors from the first
    view.
    Parameters
    ----------
    out1
    out2
    triangle_1
    triangle_2
    img1
    img2
    face_coords

    Returns
    -------

    """
    images = []
    x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr, dist = \
        find_correspondence_in_feature_space_point_image(out1,
                                                         out2,
                                                         point_image1,
                                                         point_image2)
    dist = dist.data.cpu().numpy()
    valid_indices = np.where(dist > 0.7)[0]

    images.append(img1)
    images.append(img2)

    colors = img2[x2_f[valid_indices], y2_f[valid_indices]]

    temp_image = img1.copy()
    temp_image[x1_org, y1_org] = 0.0
    temp_image[x1_org[valid_indices], y1_org[valid_indices]] = colors
    images.append(temp_image)
    return images

# def visualize_dense_correspondence_point_image(out1, out2, triangle_1, triangle_2, point_image1, point_image2, img1, img2, face_coords, viz=False):
#     """
#     Here we want to visualize dense correspondence. We start by coloring the mesh face centers
#     with coordinate based colors. Then for the first view, we take all foreground pixels and
#     find their correspondence with the second view using similarity in feature space. Then we
#     color the corresponding pixels in the second view by borrowing the colors from the first
#     view.
#     Parameters
#     ----------
#     out1
#     out2
#     triangle_1
#     triangle_2
#     img1
#     img2
#     face_coords
#
#     Returns
#     -------
#
#     """
#     images = []
#     x1_org, y1_org, x2_org, y2_org, x2_f, y2_f, x1_corr, y1_corr, x2_corr, y2_corr = \
#         find_correspondence_in_feature_space_point_image(out1,
#                                                          out2,
#                                                          point_image1,
#                                                          point_image2)
#
#     # First view, coordinate color maps for all foreground pixels
#     if viz:
#         fig, ax = plt.subplots(1, 3, figsize=(6, 1), dpi=200)
#     # fig.tight_layout()
#     colors1 = face_coords[triangle_1[x1_org, y1_org]]
#     temp_image = img1.copy()
#
#     colors1 = colors1 - colors1.min(0)
#     colors1 = colors1 / colors1.max(0)
#
#     temp_image[x1_org, y1_org] = colors1
#     images.append(temp_image)
#     if viz:
#         ax[0].imshow(temp_image)
#
#     # Second image, visualize the coordinate colors for all foreground pixels
#     colors1 = face_coords[triangle_2[x2_org, y2_org]]
#     temp_image = img2.copy()
#
#     colors1 = colors1 - colors1.min(0)
#     colors1 = colors1 / colors1.max(0)
#
#     temp_image[x2_org, y2_org] = colors1
#     images.append(temp_image)
#     if viz:
#         ax[1].imshow(temp_image)
#
#     # second image, correspondence maps for all forground pixels in the first image
#     colors1 = face_coords[triangle_1[x1_org, y1_org]]
#     colors1 = colors1 - colors1.min(0)
#     colors1 = colors1 / colors1.max(0)
#
#     temp_image = img2.copy()
#     temp_image[x2_f, y2_f] = colors1
#     images.append(temp_image)
#     if viz:
#         ax[2].imshow(temp_image)
#
#
#     # First image, coordinate color map for pixels that are in correspondence between two images
#     # colors1 = face_coords[triangle_1[x1_corr, y1_corr]]
#     # temp_image = img1.copy()
#     #
#     # colors1 = colors1 - colors1.min(0)
#     # colors1 = colors1 / colors1.max(0)
#     #
#     # temp_image[x1_corr, y1_corr] = colors1
#     # images.append(temp_image)
#     # ax[3].imshow(temp_image)
#
#     # Second image, ground truth coordinate color maps for pixels that are in correspondence between two images
#     # temp_image = img2.copy()
#     # temp_image[x2_corr, y2_corr] = colors1
#     # images.append(temp_image)
#     # ax[4].imshow(temp_image)
#     if viz:
#         for a in ax:
#             a.axis("off")
#
#     # plt.savefig("results/heatmap_all.png")
#     return images


def visualize_dense_correspondence_both_size(out1, out2, tri1, tri2, p_image1, p_image2, gray_image1, gray_image2, path=None, viz=False, second_side=False):
    mesh = prepare_mesh(path, clean=False)
    face_coords = mesh.triangles_center

    images12 = visualize_dense_correspondence_point_image_backward(out1,
                                                          out2,
                                                          tri1,
                                                          tri2,
                                                          p_image1,
                                                          p_image2,
                                                          gray_image1,
                                                          gray_image2,
                                                          face_coords=face_coords,
                                                          viz=viz)
    images12_texture = visualize_dense_correspondence_point_image_backward_texture(out1,
                                                          out2,
                                                          tri1,
                                                          tri2,
                                                          p_image1,
                                                          p_image2,
                                                          gray_image1,
                                                          gray_image2,
                                                          face_coords=face_coords,
                                                          viz=viz)

    if second_side:
        images21 = visualize_dense_correspondence_point_image_backward(out2,
                                                              out1,
                                                              tri2,
                                                              tri1,
                                                              p_image2,
                                                              p_image1,
                                                              gray_image2,
                                                              gray_image1,
                                                              face_coords=face_coords,
                                                              viz=viz)
        images21_texture = visualize_dense_correspondence_point_image_backward_texture(out2,
                                                                               out1,
                                                                               tri2,
                                                                               tri1,
                                                                               p_image2,
                                                                               p_image1,
                                                                               gray_image2,
                                                                               gray_image1,
                                                                               face_coords=face_coords,
                                                                               viz=viz)
        return images12_texture + [images21_texture[-1]] + images12 + [images21[-1]]
    else:
        return images12


def visualize_prune_dense_correspondence_both_size(out1, out2, tri1, tri2, p_image1, p_image2, gray_image1, gray_image2, path=None, viz=False, second_side=False):
    mesh = prepare_mesh(path, clean=False)
    face_coords = mesh.triangles_center

    images12 = prune_outliers_dense_correspondence(out1,
                                                          out2,
                                                          tri1,
                                                          tri2,
                                                          p_image1,
                                                          p_image2,
                                                          gray_image1,
                                                          gray_image2,
                                                          face_coords=face_coords,
                                                          viz=viz)
    if second_side:
        images21 = prune_outliers_dense_correspondence(out2,
                                                              out1,
                                                              tri2,
                                                              tri1,
                                                              p_image2,
                                                              p_image1,
                                                              gray_image2,
                                                              gray_image1,
                                                              face_coords=face_coords,
                                                              viz=viz)
        return images12, images21
    else:
        return images12


def precision_recall(out1, out2, p_image1, p_image2, img1, img2, thres=5, viz=False):
    """

    Parameters
    ----------
    out1
    out2
    p_image1
    p_image2
    img1
    img2
    thres

    Returns
    -------

    """
    # TODO precision revall curve
    # TODO both side of precision revall curve.

    x1, y1, x2, y2, x2_f, y2_f, x2_org, y2_org, dist = find_correspondence_in_feature_space_eval_metric(out1,
                                                                                                        out2,
                                                                                                        p_image1,
                                                                                                        p_image2)

    coord2 = np.stack([x2, y2], 1)
    coord2_pred = np.stack([x2_f, y2_f], 1)
    if viz:
        plt.figure()
        _ = draw_all_lines(img1, img2, y1, x1, y2_f, x2_f, N=5)
    result = {}
    for i in range(1, 5):
        result[thres * i] =(np.linalg.norm(coord2 - coord2_pred, axis=1, ord=2) < thres * i).mean()
    return result
