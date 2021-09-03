import os
import json
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image

import matplotlib.pyplot as plt

mask_dir = 'data/water_mask'
size = 100 # Original size of the environment. The mask may need to be resized to fit this environment
nms_dist = 5

equations = []
img_paths = os.listdir(mask_dir)
for img_path in img_paths:
    mask = np.array(Image.open(os.path.join(mask_dir, img_path)))
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    u_scale = size / mask.shape[1]
    v_scale = size / mask.shape[0]
    # Edges are in black. 
    edge_points = np.where(mask == 0) 
    vs, us = edge_points
    vs = vs * v_scale
    us = us * u_scale
    hull = ConvexHull(np.stack([us, vs], axis=1))
    # Remove vertices that are too close to others.
    # This can reduce the number of equations representing the convex hull.
    vertices_simplified = []
    for ind in hull.vertices:
        u, v = us[ind], vs[ind]
        too_close = False
        for vertices in vertices_simplified:
            if (vertices[0] - u) ** 2 + (vertices[1] - v) ** 2 < nms_dist ** 2:
                too_close = True 
        if not too_close:
            vertices_simplified.append([u, v])
    vertices_simplified = np.array(vertices_simplified)
    hull = ConvexHull(vertices_simplified)
    # The equations are [a, b, c] such that ax + by <= c, representing each facet.
    hull_equations = np.copy(hull.equations) # The original representation is ax + by + c <= 0
    hull_equations[:, 2] = -hull_equations[:, 2] # Convert to ax + by <= c
    equations.append(hull_equations.tolist())

json.dump(equations, open('mountains.json', 'w'), indent=4)