import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

class DataAugmentor:

	"""
    A class used for data augmentation (partially taken from : https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/)

    Attributes
    ----------
	batch : tf.Tensor, optional
		The batch to augment
	batchSize: int
		The batch size
	seed: int, optional
		Random seed

    Methods
    -------
    flip
        Flip Augmentation
	color 
		Color Augmentation
	gaussian
		Gaussian Noise
	brightness
		Custom Brightness Augmentation
	zoom
		Crop Augmentation
	kerasAug
		Inbuilt Keras Augmentations
	augment
		Wrapper Augmentation Function
    """

	def __init__(self, batch=None, batchSize=50, seed=0):
		if batch is not None:
			self.dataset = batch
		self.seed = seed
		tf.random.set_seed(self.seed)
		np.random.seed(self.seed)
		self.batchSize = batchSize

	def flip(self, x: tf.Tensor) -> tf.Tensor:
		"""Flip augmentation

		Args:
		    x: Image to flip

		Returns:
		    Augmented image
		"""
		x = tf.image.random_flip_left_right(x, seed=self.seed)

		return x

	def color(self, x: tf.Tensor) -> tf.Tensor:
		"""Color augmentation

		Args:
		    x: Image

		Returns:
		    Augmented image
		# """
		x = tf.image.random_hue(x, 0.05, seed=self.seed)
		x = tf.image.random_saturation(x, 0.6, 1.2, seed=self.seed)
		x = tf.image.random_brightness(x, 0.05, seed=self.seed)
		x = tf.image.random_contrast(x, 0.7, 1.0, seed=self.seed)

		return x

	def gaussian(self, x: tf.Tensor) -> tf.Tensor:

		mean = tf.keras.backend.mean(x)
		std = tf.keras.backend.std(x)
		max_ = tf.keras.backend.max(x)
		min_ = tf.keras.backend.min(x)
		ptp = max_ - min_
		noise = tf.random.normal(shape=tf.shape(x), mean=0, stddev=0.3*self.var, dtype=tf.float32, seed=self.seed)
		# noise_img = tf.clip_by_value(((x - mean)/std + noise)*std + mean, 
		# 	clip_value_min = min_, clip_value_max=max_)
		noise_img = x+noise

		return noise_img

	def brightness(self, x: tf.Tensor) -> tf.Tensor:

		max_ = tf.keras.backend.max(x)
		min_ = tf.keras.backend.min(x)
		brightness_val = 0.1*np.random.random_sample() - 0.05
		noise = tf.constant(brightness_val, shape=x.shape)
		noise_img = x+noise
		noise_img = tf.clip_by_value(x, 
			clip_value_min = min_, clip_value_max=max_)
		return noise_img

	def zoom(self, x: tf.Tensor) -> tf.Tensor:
		"""Zoom augmentation

		Args:
		    x: Image

		Returns:
		    Augmented image
		"""
		
		# Generate 20 crop settings, ranging from a 1% to 20% crop.
		scales = list(np.arange(0.85, 1.0, 0.01))
		boxes = np.zeros((len(scales), 4))

		for i, scale in enumerate(scales):
		    x1 = y1 = 0.5 - (0.5 * scale)
		    x2 = y2 = 0.5 + (0.5 * scale)
		    boxes[i] = [x1, y1, x2, y2]

		def random_crop(img):
		    # Create different crops for an image
		    crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(x.shape[0], x.shape[1]))
		    # Return a random crop
		    return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32, seed=self.seed)]


		choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32, seed=self.seed)

		# Only apply cropping 50% of the time
		return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

	def kerasAug(self, x: tf.Tensor) -> tf.Tensor:

		datagen = tf.keras.preprocessing.image.ImageDataGenerator(
		    rotation_range=2,
		    width_shift_range=0,
		    height_shift_range=0,
		    horizontal_flip=False,
		    shear_range = 0,
		    fill_mode='nearest',
		    dtype = tf.float32)
		
		return datagen.flow(x, batch_size=self.batchSize, shuffle=False, seed=self.seed).next()


	def augment(self, batch=None): 
		if batch is not None:
			self.dataset = batch
		self.dataset = tf.data.Dataset.from_tensor_slices(self.dataset.numpy())

		# Add augmentations
		augmentations = [self.flip, self.color, self.zoom]

		# Add the augmentations to the dataset
		for f in augmentations:
			# Apply the augmentation, run 4 jobs in parallel.
			self.dataset = self.dataset.map(f)

		self.dataset = next(iter(self.dataset.batch(self.batchSize)))
		
		return(self.dataset)




