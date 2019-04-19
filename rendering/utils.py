import math
import numpy as np
import cv2
from tqdm import tqdm
import random
from scipy.linalg import expm, norm
import os
from rendering.renderer import Renderer


def compute_rotation_from_vertex(vertex):
    """Compute rotation matrix from viewpoint vertex """
    up = [0, 0, 1]
    if vertex[0] == 0 and vertex[1] == 0 and vertex[2] != 0:    # 这是在z轴上啊
        up = [-1, 0, 0]
    rot = np.zeros((3, 3))
    rot[:, 2] = -vertex / norm(vertex)  # View direction towards origin
    rot[:, 0] = np.cross(rot[:, 2], up)    # 叉乘 获得相机的右边方向向量
    rot[:, 0] /= norm(rot[:, 0])
    rot[:, 1] = np.cross(rot[:, 0], -rot[:, 2])
    return rot.T


def create_pose(vertex, scale=0, angle_deg=0):
    """Compute rotation matrix from viewpoint vertex and inplane rotation """
    rot = compute_rotation_from_vertex(vertex)

    rodriguez = np.asarray([0, 0, 1]) * (angle_deg * math.pi / 180.0)
    angle_axis = expm(np.cross(np.eye(3), rodriguez))
    transform = np.eye(4)
    transform[0:3, 0:3] = np.matmul(angle_axis, rot)
    transform[0:3, 3] = [0, 0, scale]
    return transform


def precompute_projections(coco_images_path, image_files, sequence, views, inplanes, cam, model3D):
    """Precomputes the projection information needed for 6D pose construction

    # Arguments
        views: List of 3D viewpoint positions
        inplanes: List of inplane angles in degrees
        cam: Intrinsics to use for translation estimation
        model3D: Model3D instance
    """
    save_path = os.path.join("output", str(sequence))    # save the rendered images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f = open(os.path.join(save_path, "gt.txt"), 'w')

    w, h = 640, 480
    count = 0    # 图片计数
    ren = Renderer((w, h), cam)
    data = []
    if model3D.vertices is None:
        return data

    for v in tqdm(range(len(views))):
        data.append([])

        for index, i in enumerate(inplanes):
            pose = create_pose(views[v], angle_deg=i)
            pose[:3, 3] = [0, 0, 0.5]  # zr = 0.5

            # Render object and extract tight 2D bbox and projected 2D centroid
            ren.clear()
            ren.draw_model(model3D, pose)
            col, dep = ren.finish()
            ys_original, xs_original = np.nonzero(dep > 0)
            xs_min = xs_original.min()
            ys_min = ys_original.min()
            xs_max = xs_original.max()
            ys_max = ys_original.max()
            rows, cols = (480, 640)
            for _ in range(1):    # number of images for every view and inplane generated
                base_img = cv2.imread(os.path.join(coco_images_path, random.choice(image_files)))
                base_img = cv2.resize(base_img, (640, 480))

                col_, dep_ = col.copy(), dep.copy()
                x_minus = -random.randint(0, xs_min)/2
                y_minus = -random.randint(0, ys_min)/2
                x_plus = random.randint(0, cols-xs_max)/2
                y_plus = random.randint(0, rows-ys_max)/2

                scale = random.uniform(0.75, 1)
                M = np.array([[scale, 0, random.choice([x_minus, x_plus])],
                              [0, scale, random.choice([y_minus, y_plus])]], dtype=np.float)
                col_ = cv2.warpAffine(col_, M, (cols, rows), borderValue=0)
                dep_ = cv2.warpAffine(dep_, M, (cols, rows), borderValue=0)
                ys, xs = np.nonzero(dep_ > 0)  # 通过深度图来得出bbox
                base_img[dep_ > 0] = 0

                base_img += np.array(col_ * 255, dtype=np.uint8)

                obj_bb = [xs.min(), ys.min(), xs.max()-xs.min(), ys.max()-ys.min()]
                content = [str(count).zfill(4) + '.jpg', sequence, *obj_bb, v, index]
                f.write(' '.join(list(map(str, content))) + '\n')

                cv2.imwrite(os.path.join(save_path, str(count).zfill(4) + '.jpg'), base_img)
                count += 1
    f.close()
    return data


