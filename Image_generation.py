from _3DMM import _3DMM
import cv2
import numpy as np
from scipy.spatial import Delaunay
import mesh_ops
import image_ops


def perform_saturation(x, tau):
    x = x - np.mean(x)
    mad = np.mean(np.absolute(x - np.mean(x)))
    x = mesh_ops.clamp(x / (2 * mad), -tau, tau)
    return x


def generate_3channel(data_3dmm, render_full):
    # Fixed params
    perc = 0.11
    resize_dim = 256
    tau = 2
    deg_tol = 5

    _3DMM_obj = _3DMM()

    S = data_3dmm['S']
    R = data_3dmm['R']
    T = data_3dmm['T']
    defShape = data_3dmm['defShape']

    angles = mesh_ops.rotationMatrixToEulerAngles(R)
    angles = (180/3.14) * angles

    print(angles)

    if angles[0] > deg_tol:
        angles[0] = deg_tol
    elif angles[0] < -deg_tol:
        angles[0] = -deg_tol
    if angles[1] > deg_tol:
        angles[1] = deg_tol
    elif angles[1] < -deg_tol:
        angles[1] = -deg_tol
    angles = (3.14/180) * angles

    R = mesh_ops.eulerAnglesToRotationMatrix(angles)

    # Project the deformed shape and build the full resolution model
    # Use the frontal reference projection for consistency
    # camfile = h5py.File('data3dmm/ref-proj-front-2.mat', 'r')
    # S = np.transpose(np.array(camfile['S']))
    # R = np.transpose(np.array(camfile['R']))
    # T = np.transpose(np.array(camfile['t']))

    defShape_2D = np.transpose(_3DMM_obj.getProjectedVertex(defShape, S, R, T))

    if render_full:
        defShape = mesh_ops.fullRes3DMM(defShape, defShape_2D, data_3dmm['visIdx'])
        defShape_2D = np.transpose(_3DMM_obj.getProjectedVertex(defShape, S, R, T))

    print('Compute Triangulation')
    tri = Delaunay(defShape[:, :2])
    faces = np.asarray(tri.simplices, dtype=int)

    print('Compute Curvatures')
    Umin, Umax, Cmin, Cmax, Cmean, Cgauss, Normal = mesh_ops.compute_curvature(defShape, faces)
    colors_curv = perform_saturation(np.abs(Cmin)+np.abs(Cmax), tau)

    print('Compute Normals')
    normals, normalf = mesh_ops.compute_normals(defShape, faces)
    normals = normals.T

    azimuth, elevation, r = mesh_ops.cart2sph(normals[:, 0], normals[:, 1], normals[:, 2])

    print('Building Image')
    elev_im = image_ops.build_depth_adapt(defShape_2D, elevation.T, 640, 480)
    azimuth_im = image_ops.build_depth_adapt(defShape_2D, colors_curv.T, 640, 480)
    depth_im = image_ops.build_depth_adapt(defShape_2D, defShape[:, 2], 640, 480)
    final_image = np.stack((depth_im, azimuth_im, elev_im), axis=-1)

    # Crop to remove background
    minx = np.amin(defShape_2D[:, 0])
    maxx = np.amax(defShape_2D[:, 0])
    miny = np.amin(defShape_2D[:, 1])
    maxy = np.amax(defShape_2D[:, 1])

    w = (maxx - minx)
    h = (maxy - miny)
    bbox_e = (max(minx-(h*perc), 1), max(miny-(h*perc), 1), w+(h*perc*2), h+(h*perc*2))

    final_image2 = final_image[int(bbox_e[1]):int(bbox_e[1]+bbox_e[3]), int(bbox_e[0]):int(bbox_e[0]+bbox_e[2]), :]
    final_image2 = image_ops.add_black_border(final_image2)

    final_image2 = cv2.resize(final_image2, dsize=(resize_dim, resize_dim), interpolation=cv2.INTER_CUBIC)

    print('Image Completed')
    return final_image2

