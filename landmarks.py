import face_alignment
import image_ops
import cv2

class LMDetector(object):
	def __init__(self):
		self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')

	def get3DLandmarks_file(self, im):

		im2 = image_ops.add_black_border(im)
		im2 = cv2.resize(im2, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

		preds = self.fa.get_landmarks(im2)

		return preds[0], im2

	def get3DLandmarks_camera(self, im):

		# Face Detection and resize of the image
		bbox = self.fa.face_detector.detect_from_image(im[..., ::-1].copy())
		minx = bbox[0][0]
		maxx = bbox[0][2]
		miny = bbox[0][1]
		maxy = bbox[0][3]
		w = (maxx - minx)
		h = (maxy - miny)
		bbox_e = (max(minx - (h * 0.25), 1), max(miny - (h * 0.25), 1), w + (h * 0.25 * 2), h + (h * 0.25 * 2))
		im2 = im[int(bbox_e[1]):int(bbox_e[1] + bbox_e[3]), int(bbox_e[0]):int(bbox_e[0] + bbox_e[2]), :]
		im2 = image_ops.add_black_border(im2)
		im2 = cv2.resize(im2, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)

		preds = self.fa.get_landmarks(im2)

		return preds[0], im2
