# Deep 3D morphable model refinement via progressive growing of conditional Generative Adversarial Networks

This page contains end-to-end demo code to recover accurate and detailed reconstructions of the facial shape from an unconstrained 2D face image. For a given input image, it fits a 3D Morhable Face Model [2] to the face image and then uses a CGAN to refine the reconstruction by adding local fine-grained details. The code outputs a standard obj file (without texture). The full approach is described in our paper [1].

## Library Requirements

* cmake
* dlib
* face-alignment
* opencv-python
* scipy==1.1.0
* scikit-image
* tensorflow

Dependencies can be installed by running `pip install -r requirements.txt`

## Data Requirements

Use the following link to download the [3DMM data](https://drive.google.com/a/unifi.it/file/d/12ull7YHxsqEvF4OlllOc8kneS9h4fI7y/view?usp=sharing) and the [CGAN weights](https://drive.google.com/a/unifi.it/file/d/1FaGyOygkYbL8UpUYgAv2Hh7h-KergaDl/view?usp=sharing).

The *data3dmm* foler must be unzipped in the root folder; the *.pb* file must be also placed in the root folder.

## Usage

In the root folder, the demo script can be run with the following: `python main.py <--im_path path/to/image> <--use_camera>`

If `--im_path` is not specified, the default image is used. Test images must be placed into *testdata* folder. If a webcam is available, `--use_camera` can be used to capture a live frame.

## References

[1] Galteri, Leonardo, et al. "Deep 3D morphable model refinement via progressive growing of conditional Generative Adversarial Networks." Computer Vision and Image Understanding 185 (2019): 31-42.
[2] Ferrari, Claudio, et al. "A dictionary learning-based 3D morphable shape model." IEEE Transactions on Multimedia 19.12 (2017): 2666-2679.
