import numpy as np
from scipy.interpolate import LinearNDInterpolator
from skimage import draw


def build_depth_adapt(proj_shape, z3d, w, h):
    step = 1
    # Scale data
    z3d = np.interp(z3d, (z3d.min(), z3d.max()), (0, 255))
    # Build grid
    Xsampling = np.arange(1, w, step, dtype='float')
    Ysampling = np.arange(1, h, step, dtype='float')
    x_grid, y_grid = np.meshgrid(Xsampling, Ysampling)

    Fz = LinearNDInterpolator(proj_shape, z3d)

    depth_im = Fz(x_grid, y_grid)

    mask3d = np.isnan(depth_im)

    depth_im[mask3d] = 0
    depth_im = np.uint8(depth_im)
    return depth_im


def add_black_border(img):
    dim = img.shape
    offset = int(np.floor(np.abs((dim[0]-dim[1])/2)))

    if dim[0] < dim[1]:
        border = np.zeros((offset, dim[1], img.shape[2]), dtype=np.uint8)
        img = np.vstack((border, img, border))
    else:
        border = np.zeros((dim[0], offset, img.shape[2]), dtype=np.uint8)
        img = np.hstack((border, img, border))
    return np.uint8(img)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask