from _3DMM import _3DMM
import numpy as np
import h5py
import numpy.matlib as npm
import scipy.io


def fit_3dmm(lm):
    # Fixed Params
    _lambda = 0.15
    rounds = 1
    r = 3.8
    c_dist = 400

    # Load landmarks, delete the reduntant keypoints
    # lmfile = h5py.File('testdata/lm.mat', 'r')
    # lm = np.transpose(np.array(lmfile['lm']))

    lm = lm[:, :2]
    lm = np.delete(lm, 64, axis=0)
    lm = np.delete(lm, 60, axis=0)

    # Load 3D data
    avgfile = h5py.File('data3dmm/avgModel_bh_1779_NE_mediumBound.mat', 'r')
    avg_model = np.transpose(np.array(avgfile['avgModel']))
    idx_landmarks_3D = np.transpose(np.array(avgfile["idxLandmarks3D"]))
    idx_landmarks_3D -= 1
    landmarks_3D = np.transpose(np.array(avgfile["landmarks3D"]))

    # Center to zero
    baric_3dmm = np.mean(avg_model, axis=0)
    avg_model = avg_model - npm.repmat(baric_3dmm, avg_model.shape[0], 1)
    landmarks_3D = landmarks_3D - npm.repmat(baric_3dmm, landmarks_3D.shape[0], 1)

    # Load 3DMM params and dictionary
    componentsfile = h5py.File('data3dmm/components_DL_300_1779.mat', 'r')
    Components = np.transpose(np.array(componentsfile["Components"]))
    Weights = np.transpose(np.array(componentsfile["Weights"]))
    Components_res = np.transpose(np.array(componentsfile["Components_res"]))
    print('Data Loaded.')

    # Create 3DMM object
    _3DMM_obj = _3DMM()

    # Fit the 3DMM
    result = _3DMM_obj.opt_3DMM_fast(Weights, Components, Components_res,
                                     landmarks_3D, idx_landmarks_3D, lm, avg_model, _lambda, rounds, r, c_dist)
    print('3DMM Fitting Completed')

    return result

