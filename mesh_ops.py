import numpy.matlib as npm
from scipy.interpolate import LinearNDInterpolator
import cv2
import numpy as np
from scipy.spatial import Delaunay
from skimage import draw
import scipy.sparse as sps
import math
import h5py


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def clamp(n, smallest, largest):
    return np.maximum(smallest, np.minimum(n, largest))


def cart2sph(x, y, z):
    hypotxy = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt(hypotxy + z ** 2)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    return az, elev, r


def _2linear_vects(X, Y):
    coords = np.empty((1, 2))
    for i in range(X.shape[0]):
        new_cord = np.array([X[i, 0], Y[i, 0]]).reshape(1, 2, order='F')
        coords = np.row_stack((coords, new_cord))
    coords = np.delete(coords, (0), axis=0)
    return coords


def check_face_vertex(V, F):

    if V.shape[0] > V.shape[1]:
        V = V.T
    if F.shape[0] > F.shape[1]:
        F = F.T
    return V, F


def fullRes3DMM(defShape, proj_shape, visIdx):
    grid_step = 1
    max_sh = (np.amax(proj_shape, axis=0))
    max_sh = np.reshape(max_sh,(1,2),order='F')
    min_sh = np.amin(proj_shape, axis=0)
    min_sh = np.reshape(min_sh, (1,2), order='F')
    Xsampling = np.arange(min_sh[0,0], max_sh[0,0], grid_step, dtype='float')
    Ysampling = np.arange(max_sh[0,1], min_sh[0,1], -grid_step, dtype='float')
    # round the float numbers
    Xsampling = np.around(Xsampling)
    Ysampling = np.around(Ysampling)

    x_grid, y_grid = np.meshgrid(Xsampling, Ysampling)

    index = np.array(visIdx, dtype=np.intp)
    X = proj_shape[index, 0]
    Y = proj_shape[index, 1]
    coords = _2linear_vects(X, Y)
    Fx = LinearNDInterpolator(coords, defShape[index, 0])
    Fy = LinearNDInterpolator(coords, defShape[index, 1])
    Fz = LinearNDInterpolator(coords, defShape[index, 2])

    x = Fx(x_grid.flatten(order='F'), y_grid.flatten(order='F'))
    y = Fy(x_grid.flatten(order='F'), y_grid.flatten(order='F'))
    z = Fz(x_grid.flatten(order='F'), y_grid.flatten(order='F'))

    x = x[~np.isnan(x).any(axis=1)]
    y = y[~np.isnan(y).any(axis=1)]
    z = z[~np.isnan(z).any(axis=1)]
    mod3d = np.dstack([x, y, z])
    mod3d = np.reshape(mod3d,(mod3d.shape[0], 3))
    print("Dense Resampling Completed")
    return mod3d


def compute_normals(vertex, faces):

    vertex, faces = check_face_vertex(vertex, faces)

    nface = faces.shape[1]
    nvert = vertex.shape[1]

    normals = np.zeros((3, nvert))

    # Unit normals to the faces
    normalf = np.cross((vertex[:, faces[1, :]] - vertex[:, faces[0, :]]).T,
                       (vertex[:, faces[2, :]] - vertex[:, faces[0, :]]).T).T

    d = np.sum(normalf**2, axis=0)
    d = np.sqrt(d)
    d[d < np.finfo(float).eps] = 1
    normalf = normalf / npm.repmat(d, 3, 1)

    # Unit normals to the vertex
    for i in range(nface):
        f = faces[:, i]
        for j in range(3):
            normals[:, f[j]] = normals[:, f[j]] + normalf[:, i]

    # Normalize
    d = np.sqrt(np.sum(normals ** 2, axis=0))
    normals = normals / npm.repmat(d, 3, 1)

    # Enforce the normals outwards
    v = vertex - npm.repmat(np.mean(vertex, axis=0), 3, 1)
    s = np.sum(v * normals, axis=1)
    if np.sum(s > 0) < np.sum(s < 0):
        normals = -normals
        normalf = -normalf

    return normals, normalf


def triangulation2adjancency(face):
    if face.shape[0] > face.shape[1]:
        face = face.T
    f = face.T

    a1 = np.concatenate((f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]), axis=0)
    a2 = np.concatenate((f[:, 1], f[:, 2], f[:, 0], f[:, 2], f[:, 0], f[:, 1]), axis=0)

    A = sps.csr_matrix((np.tile(1.0, a1.shape[0]), (a1, a2)))

    A = (A > 0).astype(float)
    return A


def perform_mesh_smoothing(face, vertex, f, naver):

    face, vertex = check_face_vertex(face, vertex)
    # if f.shape[1] > 1:
    #     for i in range(f.shape[1]):
    #         f[:, i] = perform_mesh_smoothing(face, vertex, f[:, i], naver)
    #     return

    n = np.max(face.flatten('F'))
    n = n+1
    W = triangulation2adjancency(face)
    W += sps.eye(n) # account for the 0-based indexing
    D = sps.spdiags(np.asarray(np.power(np.sum(W, axis=1), -1)).squeeze(), 0, n, n)
    W = D*W

    for k in range(naver):
        f = W * f

    return f


def compute_curvature(vertex, faces):

    orient = 1
    naver = 6

    # matlab = h5py.File('testdata/defShape.mat', 'r')
    # vertex = np.transpose(np.array(matlab['V']))
    # faces = np.transpose(np.array(matlab['F']))
    # faces = faces.T.astype(np.int)
    # faces = faces - 1

    V, F = check_face_vertex(vertex, faces)
    m, n = F.shape[1], V.shape[1]

    i, j, s = np.hstack([F[0, :], F[1, :], F[2, :]]), np.hstack([F[1, :], F[2, :], F[0, :]]), np.hstack(
        [range(m), range(m), range(m)])

    I, idx = np.unique(np.vstack([i, j]).transpose(), axis=0, return_index=True)
    i, j, s = i[idx].astype(np.int), j[idx].astype(np.int), s[idx].astype(np.int)
    s += 1
    S = sps.csr_matrix((s, (i, j)))
    dnz = np.nonzero(S)
    rnz = np.nonzero(S.transpose())
    s2 = S[dnz[0], dnz[1]].A1
    s1 = S[rnz[1], rnz[0]].A1

    I = np.nonzero(np.logical_and(s1 > 0, s2 > 0))
    s1 -= 1
    s2 -= 1

    E = np.stack([s1, s2]).transpose()
    i = dnz[1][I]
    j = dnz[0][I]

    I = np.argwhere(i < j).squeeze()
    E = E[I, :]

    i = i[I]
    j = j[I]

    ne = i.shape[0]

    e = V[:, j] - V[:, i]
    d = np.sqrt(np.sum(e ** 2, 0))
    e = e / d
    d = d / np.mean(d)

    # Normals of faces
    normal, normalf = compute_normals(V, F)
    # inner product of normals
    dp = np.sum(normalf[:, E[:, 0]]*normalf[:, E[:, 1]], axis=0)
    # angle un-signed
    beta = np.arccos(clamp(dp, -1, 1))
    # sign
    cp = np.cross(normalf[:, E[:, 0]].T, normalf[:, E[:, 1]].T).T
    si = orient * np.sign(np.sum(cp * e, axis=0))
    # angle signed
    beta = beta * si
    # tensors
    T = np.zeros((3, 3, ne))

    for x in range(3):
        for y in range(x+1):
            T[x, y, :] = np.reshape(e[x, :] * e[y, :], (1, 1, ne))
            T[y, x, :] = T[x, y, :]

    T = T * np.tile(np.reshape(d * beta, (1, 1, ne)), (3, 3, 1))

    # do pooling on vertices
    Tv = np.zeros((3, 3, n))
    w = np.zeros((1, 1, n))

    for k in range(ne):
        Tv[:, :, i[k]] = Tv[:, :, i[k]] + T[:, :, k]
        Tv[:, :, j[k]] = Tv[:, :, j[k]] + T[:, :, k]
        w[:, :, i[k]] = w[:, :, i[k]] + 1
        w[:, :, j[k]] = w[:, :, j[k]] + 1

    w[w < np.finfo(float).eps] = 1
    Tv = Tv / np.tile(w, (3, 3, 1))

    # do averaging to smooth the field
    for x in range(3):
        for y in range(3):
            a = Tv[x, y, :]
            a = perform_mesh_smoothing(F, V, a.squeeze(), naver)
            Tv[x, y, :] = np.reshape(a, (1, 1, n))

    # extract eigenvectors and eigenvalues
    U = np.zeros((3, 3, n))
    D = np.zeros((3, n))

    for k in range(n):
        eigval, eigvec = np.linalg.eig(Tv[:, :, k])
        eigval = np.real(eigval)
        I = np.argsort(np.abs(eigval))
        D[:, k] = eigval[I]
        U[:, :, k] = np.real(eigvec[:, I])

    Umin = U[:, 2, :].squeeze()
    Umax = U[:, 1, :].squeeze()
    Cmin = D[1, :].T
    Cmax = D[2, :].T

    Normal = U[:, 0, :].squeeze()
    Cmean = (Cmin + Cmax) / 2
    Cgauss = Cmin * Cmax

    # enforce min < max
    I = np.nonzero(Cmin > Cmax)
    Cmin1 = Cmin.copy()
    Umin1 = Umin
    Cmin[I] = Cmax[I]
    Cmax[I] = Cmin1[I]
    Umin[:, I] = Umax[:, I]; Umax[:, I] = Umin1[:, I]

    normal, normalf = compute_normals(V, F)
    s = np.sign(np.sum(Normal * normal, axis=0))
    Normal = Normal * npm.repmat(s, 3, 1)

    return Umin, Umax, Cmin, Cmax, Cmean, Cgauss, Normal


def mesh_from_depth(im, bound_size):

    depth = im[:, :, 0]
    im_size = depth.shape[0]

    x_sampling = np.arange(0, depth.shape[0], 1, dtype='float')
    y_sampling = np.arange(depth.shape[1], 0, -1, dtype='float')
    x, y = np.meshgrid(x_sampling, y_sampling)
    thresh, mask = cv2.threshold(depth, 40, 255, 0)

    # TODO contour: cut the border of model
    # boundary, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # boundary = np.asarray(boundary).squeeze()
    # boundary[boundary[:, 0] < np.round(im_size / 2), 0] -= bound_size
    # boundary[boundary[:, 0] > np.round(im_size / 2), 0] += bound_size
    # boundary[boundary[:, 1] < np.round(im_size / 2), 1] -= bound_size
    # boundary[boundary[:, 1] > np.round(im_size / 2), 1] += bound_size
    # mask_b = poly2mask(boundary[:, 0], boundary[:, 1], (256, 256))

    ids = np.reshape(mask, (im_size * im_size, 1), order='F')

    # Build Mesh
    vertex = np.zeros((im_size * im_size, 3))
    vertex[:, 0] = np.reshape(x.T, (im_size * im_size, 1), order='F').squeeze()
    vertex[:, 1] = np.reshape(y.T, (im_size * im_size, 1), order='F').squeeze()
    vertex[:, 2] = np.reshape(depth, (im_size * im_size, 1), order='F').squeeze()

    mesh = vertex[np.nonzero(ids), :]
    mesh = mesh[0, :, :]

    mesh[:, 2] = np.interp(mesh[:, 2], (mesh[:, 2].min(), mesh[:, 2].max()), (120, 255))

    tri = Delaunay(mesh[:, :2])

    result = {}
    result['points'] = mesh
    result['polygons'] = np.asarray(tri.simplices, dtype=int)

    print('Mesh Generated')

    return result


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def output_obj_with_faces(mesh, filename):
    f = open(filename,'w')
    f.write('g \n')
    for p in mesh['points']:
        if len(p) == 2:
            f.write('v ' + ' '.join(map(str,p[0])) + '\n')
        else:
            f.write('v ' + ' '.join(map(str,p)) + '\n')
    for p in mesh['polygons']:
        f.write("f")
        for i in p:
            f.write(" %d" % (i + 1))
        f.write("\n")
    f.write('g 1 \n')
    f.close()