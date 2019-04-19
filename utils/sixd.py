
import os
import numpy as np
import ruamel.yaml as yaml
from rendering.model import Model3D

''' The following were copied over from the 6DB toolkit'''


def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f, Loader=yaml.CLoader)
        return content


def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape((3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(info[eid]['cam_t_w2c']).reshape((3, 1))
    return info


def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts


class Frame:
    def __init__(self):
        self.nr = None
        self.color = None
        self.depth = None
        self.cam = np.identity(3)
        self.gt = []


class Benchmark:
    def __init__(self):
        self.frames = []
        self.cam = None
        self.models = {}


def load_sixd(base_path, seq, load_mesh=True):
    bench = Benchmark()
    bench.scale_to_meters = 0.001
    if os.path.exists(os.path.join(base_path, 'camera.yml')):
        cam_info = load_yaml(os.path.join(base_path, 'camera.yml'))
        bench.cam = np.eye(3)
        bench.cam[0, 0] = cam_info['fx']
        bench.cam[0, 2] = cam_info['cx']
        bench.cam[1, 1] = cam_info['fy']
        bench.cam[1, 2] = cam_info['cy']
        bench.scale_to_meters = 0.001 * cam_info['depth_scale']
    else:
        print("Could not find camera.yml. Taking camera matrix of the first frame instead.")

    model_info = load_yaml(os.path.join(base_path, 'models', 'models_info.yml'))

    for key, val in model_info.items():
        name = 'obj_{:02d}'.format(int(key))    # key: (1,2,3,4,5)
        bench.models[name] = Model3D()
        bench.models[name].diameter = val['diameter']

    if seq is None:
        return bench

    if load_mesh:
        name = 'obj_{:02d}'.format(int(seq))
        bench.models[name].load(os.path.join(base_path, 'models/' + name + '.ply'), scale=bench.scale_to_meters)

    return bench


