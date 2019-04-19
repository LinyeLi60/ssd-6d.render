"""
Script to generate train images for ssd-6d.
Usage:
    run.py [options]
    run.py (-h | --help)

Options:
    -d, --dataset=<string>   Path to SIXD dataset [default: dataset/ycb]
    -s, --sequence=<int>     Number of the sequence [default: 1]
    -h --help                Show this message and exit
"""

from docopt import docopt
import numpy as np
import os
from rendering.utils import precompute_projections
from utils.sixd import load_sixd

args = docopt(__doc__)
sixd_base = args["--dataset"]
sequence = int(args["--sequence"])
views = np.load("numpy_files/views.npy")
inplanes = np.load("numpy_files/inplanes.npy")
models = ['obj_{:02d}'.format(sequence), ]  # Overwrite model name
bench = load_sixd(sixd_base, seq=sequence)
print('Models:', models)
print('Precomputing projections for each used model...')
coco_images_path = "images"
image_files = os.listdir(coco_images_path)    # put coco images as background in this folder
model_map = bench.models  # Mapping from name to model3D instance
for model_name in models:
    m = model_map[model_name]
    m.projections = precompute_projections(coco_images_path, image_files, sequence, views, inplanes, bench.cam, m)    # TODO：看这个函数

