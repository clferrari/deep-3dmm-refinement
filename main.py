from landmarks import LMDetector
from refinement import Generator
import Fitting
import Image_generation
import scipy.misc
import mesh_ops
import scipy.io as sio
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imageio as io
import sys, os
import argparse
import shlex

def run_pipeline():
    print('Running...')
    parser = argparse.ArgumentParser(description='Fit 3DMM to a face image and refines the mesh using GAN')
    parser.add_argument('--use_camera', dest='use_camera', help='Use camera to grab frame', action='store_true')
    parser.add_argument('--im_path', dest='im_path', help='Path to test image', default='g2.png', type=str)
    args = parser.parse_args()
    use_camera = args.use_camera

    # Instantiate Landmark Detector
    lmdetector = LMDetector()

    if use_camera:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('img1', frame)
            if cv2.waitKey(1) & 0xFF == ord('y'):
                im = frame
                cv2.imwrite('testdata/test_webcam.jpg', frame)
                cv2.destroyAllWindows()
                print('Saved frame from webcam')
                break

        cap.release()
        lms, img = lmdetector.get3DLandmarks_camera(im)
    else:
        im_path = 'testdata/' + args.im_path
        im = io.imread(im_path)
        print('Reading image', im_path)
        if im.shape[2] > 3:
            im = im[..., :3]
        lms, img = lmdetector.get3DLandmarks_file(im)

    print('Detecting Landmarks')
    plt.imshow(img)
    plt.plot(lms[:, 0], lms[:, 1], 'ro')
    plt.show()

    # Fit 3DMM
    result = Fitting.fit_3dmm(lms)
    defShape = result['defShape']
    sio.savemat('testdata/defShape.mat', mdict={'mesh': defShape})
    # Generate Coarse Image
    img = Image_generation.generate_3channel(result, True)
    img = np.fliplr(img)
    # Save coarse 3-channel image
    scipy.misc.imsave('coarse.png', img)

    # Save coarse 3D mesh
    mesh_coarse = mesh_ops.mesh_from_depth(img, 1)
    mesh_ops.output_obj_with_faces(mesh_coarse, 'coarse.obj')

    # Start Refinement
    generator = Generator()
    refined = generator.refine(img)
    # Save Refined Image
    scipy.misc.imsave('refined.png', refined)
    # Output refined mesh
    mesh = mesh_ops.mesh_from_depth(refined, 1)
    mesh_ops.output_obj_with_faces(mesh, 'refined.obj')

    sio.savemat('testdata/testmesh.mat', mdict={'mesh': mesh})
    # Launch Meshlab
    os.system('/Applications/meshlab.app/Contents/MacOS/meshlab refined.obj &')


if __name__ == '__main__':
    run_pipeline()