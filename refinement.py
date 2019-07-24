import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

class Generator(object):
	def __init__(self):
		self.graph = load_graph('frozentensorflowModel.pb')
		for op in self.graph.get_operations():
			print(op.name)
		self.x = self.graph.get_tensor_by_name('prefix/z:0')			
		self.y = self.graph.get_tensor_by_name('prefix/gen/decoder/256x256/to_image/Tanh:0')
		self.sess = tf.Session(graph=self.graph)

	def refine(self, img):
			img = img / 127.5 - 1.
			out = self.sess.run(self.y, feed_dict={self.x: [img]})
			out = (out + 1) * 127.5
			return np.clip(out[0],0.,255.).astype(np.uint8)