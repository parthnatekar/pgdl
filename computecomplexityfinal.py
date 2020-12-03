"""Custom Complexity Measure based on Margin Distribution"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
import json
import pickle
import os
import time 
import sys
import random
import math
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from scipy.stats import *
sys.path.append('..')
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import silhouette_score
from math import log
import matplotlib.pyplot as plt
from augment import *
import gc
import time

class CustomComplexityFinal:

	"""
    A class used to create margin based complexity measures 

    Attributes
    ----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	rootpath : str, optional
		Path to root directory
	computeOver : int
		The number of samples over which to compute the complexity measure
	batchSize: int
		The batch size
	basename: str, optional
		basename argument of PGDL directory structure
	metric: str, optional
		Metric to use to scale margin Distribution
	augment : str, optional
		The type of augmentation to use ('standard', 'mixup', 'adverserial', 'adverserial+standard', 'mixup+standard')
	penalize : bool, optional
		Whether to penalize misclassified samples
	input_margin : bool, optional
		Whether to compute margin on input data instead of intermediate representations
	network_scale: bool, optional
		Only used for auxiliary experiments involving regularizing for network topology
	seed: int, optional
		Random seed

    """

	def __init__(self, model, ds, rootpath=None, mid=None, computeOver = 500, batchSize = 50, basename=None, metric='batch_variance', augment='standard', penalize=True, input_margin=False, network_scale = False, seed=1):
		self.model = model
		self.dataset = ds
		self.computeOver = computeOver
		if rootpath is not None:
			self.rootPath = os.path.join(rootpath, 'computed_data/pickles/{}'.format(mid))
			if not os.path.exists(self.rootPath):
				os.makedirs(self.rootPath)
		self.mid = mid
		self.basename = basename
		self.batchSize = batchSize
		self.metric = metric
		self.verbose = False
		self.norm = 'l2'
		self.penalize = penalize
		self.augment = augment
		self.input_margin = input_margin
		self.network_scale = network_scale
		self.seed=seed
 
	# ====================================== Functions for Margin Based Solution =====================================

	def computeMargins(self, top = 2):

		'''
		Fuction to compute margin distribution

		Returns
		-------
		marginDistribution : dict
			Dictionary containing lists of margin distribution for each layer

		'''

		it = iter(self.dataset.repeat(-1).shuffle(5000, seed=self.seed).batch(self.batchSize))
		marginDistribution = {}
		totalVariance = {}
		totalVarianceTensor = {}
		totalNorm = {}
		totalNormTensor = {}
		self.layers = []
		ratio_list = {}

		for l in range(len(self.model.layers)):
			c = list(self.model.get_layer(index = l).get_config().keys())
			if 'filters' in c or 'units' in c:
				self.layers.append(l)
			if len(self.layers) == 1:
				break

		if self.input_margin == True:
			self.layers = [-1]

		if self.verbose == True:
			for l, layer in enumerate(self.model.layers):
				print(self.model.get_layer(index = l).get_config())

		for i in range(self.computeOver//self.batchSize):
			batch = next(it)
			if self.augment == 'standard':
				D = DataAugmentor(batch[0], batchSize = self.batchSize)
				batch_ = (D.augment(), batch[1])
			elif self.augment == 'adverserial':
				batch_ = (self.getAdverserialBatch(batch), batch[1])
			elif self.augment == 'adverserial+standard':
				D = DataAugmentor(batch[0], batchSize = self.batchSize)
				batch_ = (D.augment(), batch[1])
				batch_ = (self.getAdverserialBatch(batch_), batch_[1])
			elif self.augment == 'mixup':
				batch_ = self.batchMixupLabelwiseLinear(batch)
			elif self.augment == 'mixup+standard':
				D = DataAugmentor(batch[0], batchSize = self.batchSize)
				batch_ = (D.augment(), batch[1])
				batch_ = self.batchMixupLabelwise(batch_)
			else:
				batch_ = batch

			for layer in self.layers:
				if self.augment is not None:
					grads, inter = self.distancefromMargin(batch_, layer+1, top)
				else:
					grads, inter = self.distancefromMargin(batch_, layer+1, top)
				try:
					marginDistribution[layer] += grads
					totalVarianceTensor[layer] = np.vstack((totalVarianceTensor[layer], np.array(inter).reshape(inter.shape[0], -1)))
				except Exception as e:
					marginDistribution[layer] = grads
					totalVarianceTensor[layer] = np.array(inter).reshape(inter.shape[0], -1)

		if self.network_scale == True:
			return marginDistribution, {}
			
		normWidth = {}
		for layer in self.layers:
			totalVariance[layer] = (trim_mean(np.var(totalVarianceTensor[layer].reshape(totalVarianceTensor[layer].shape[0], -1), axis = 0), proportiontocut=0.05))**(1/2)
			normWidth[layer] = math.sqrt(np.prod(totalVarianceTensor[layer].shape[1:]))
			marginDistribution[layer] = np.array(marginDistribution[layer])/(np.array(totalVariance[layer])+1e-7) #/np.sqrt(m_factor) #/totalVarianceTensor[layer].shape[1:][0]

		return marginDistribution, normWidth

	def distancefromMargin(self, batch, layer, top = 2):

		'''
		Fuction to calculate margin distance for a given layer

		Parameters
		----------
		batch : tf.data.Dataset()
			The batch over which to compute the margin distance. A tuple of tf.Tensor of the form (input data, labels)
		layer : int
			The layer for which to compute margin distance
		top : int, optional
			Index for which to compute margin. For example, top = 2 will compute the margins between the class with the highes and second-highest softmax scores

		Returns
		-------
		grads : list
			A list containing the scaled margin distances
		np_out : np.array
			An array containing the flattened intermediate feature vector
		'''

		if self.network_scale == True:
			batch_ = tf.ones(shape = batch[0].shape)
		else:
			batch_ = batch[0]
		with tf.GradientTape(persistent=True) as tape:
			intermediateVal = [batch_]
			tape.watch(intermediateVal)
			for l, layer_ in enumerate(self.model.layers):
				intermediateVal.append(layer_(intermediateVal[-1]))

			out_hard = tf.math.top_k(tf.nn.softmax(intermediateVal[-1], axis = 1), k = top)[1]
			top_1 = out_hard[:,top-2]
			misclassified = np.where(top_1 != batch[1])

			if self.penalize:
				top_1_og = out_hard[:,top-2]
				top_2_og = out_hard[:,top-1]
				mask  = np.array(top_1_og == batch[1]).astype(int)
				top_1 = top_1_og*mask + batch[1]*(1.- mask)
				top_2 = top_2_og*mask + top_1_og*(1.- mask)
			else:
				top_1 = out_hard[:,top-2]
				top_2 = out_hard[:,top-1]
				mask  = np.array(top_1 == batch[1]).astype(int)
				top_2 = top_2*mask + batch[1]*(1.- mask)

			index = list(range(batch[0].shape[0])) 
			index1 = np.array(list(zip(index, top_1)))
			index2 = np.array(list(zip(index, top_2)))
			preds = intermediateVal[-1]
			logit1 = tf.gather_nd(preds, tf.constant(index1, tf.int32))
			logit2 = tf.gather_nd(preds, tf.constant(index2, tf.int32))
			if self.network_scale == True:
				grad_i = tape.gradient(intermediateVal[-1], intermediateVal[layer])
				grad_diff = (np.reshape(grad_i.numpy(), (self.batchSize, -1)))
				denominator = np.linalg.norm(grad_diff, axis = 1, ord=2)
				np_out 	= np.array(intermediateVal[layer])
				print(denominator, np.mean(grad_i**2))
				return denominator, np_out, grad_diff
			else:
				grad_i = tape.gradient(logit1, intermediateVal[layer])
				grad_j = tape.gradient(logit2, intermediateVal[layer])
				numerator = tf.gather_nd(preds, tf.constant(index1, tf.int32)) - tf.gather_nd(preds, tf.constant(index2, tf.int32)).numpy()
				grad_diff = (np.reshape(grad_i.numpy(), (grad_i.numpy().shape[0], -1)) - np.reshape(grad_j.numpy(), (grad_j.numpy().shape[0], -1)))
				denominator = np.linalg.norm(grad_diff, axis = 1, ord=2)
			inf = np.linalg.norm(grad_diff, axis = 1, ord=np.inf)

			if self.penalize == False:
				numerator = np.delete(numerator, misclassified, axis = 0)
				denominator = np.delete(denominator, misclassified, axis = 0)

			if self.metric == 'spectral':
				grads = numerator
			else:
				grads = numerator/(denominator+1e-7)
			
			np_out 	= np.array(intermediateVal[layer])

		gc.collect()
		
		return list(grads), np_out


	# ====================================== Functions for Mixup Based Solution ======================================
	
	def batchMixup(self, batch, seed=1):

		'''
		Fuction to perform mixup on a batch of data

		Parameters
		----------
		batch : tf.data.Dataset()
			The batch over which to compute the margin distance. A tuple of tf.Tensor of the form (input data, labels)
		seed : int, optional
			Random seed

		Returns
		-------
		tf.tensor
			The mixed-up batch
		'''

		np.random.seed(seed)
		x = batch[0]
		mix_x = batch[0].numpy()
		for i in range(np.max(batch[1])):
			lam = (np.random.randint(0, 3, size=x[batch[1] == i].shape[0])/10)[..., None, None, None]
			mix_x[(batch[1] == i)] = x[batch[1] == i]*lam + tf.random.shuffle(x[batch[1] == i], seed=seed)*(1-lam)

		if self.augment == 'mixup':
			mix_x = mix_x[np.random.randint(0, mix_x.shape[0], size=self.batchSize//3)]
			target = batch[1].numpy()[np.random.randint(0, mix_x.shape[0], size=self.batchSize//3)]
			return (tf.convert_to_tensor(mix_x), tf.convert_to_tensor(target))
		else:
			return tf.convert_to_tensor(mix_x)

	def batchMixupLabelwiseLinear(self, batch, seed=2):


		np.random.seed(seed)
		labels = batch[1]
		sorted_indices = np.argsort(batch[1])
		sorted_labels = batch[1].numpy()[sorted_indices]
		sorted_images = batch[0].numpy()[sorted_indices]
		edges = np.array([len(sorted_labels[sorted_labels == i]) for i in range(max(sorted_labels)+1)])
		edges = [0] + list(np.cumsum(edges))
		shuffled_indices = []
		for i in range(len(edges)-1):
			# print(sorted_indices[edges[i]:edges[i+1]], sorted_labels[edges[i]:edges[i+1]])
			shuffled_indices += list(np.random.choice(list(range(edges[i], edges[i+1])), replace=False, size=edges[i+1] - edges[i]))
			# print(shuffled_indices[edges[i]:edges[i+1]])
		intrapolateImages = (sorted_images + sorted_images[shuffled_indices])/2
		return (tf.convert_to_tensor(intrapolateImages), tf.convert_to_tensor(sorted_labels))

	def batchMixupLabelwise(self, batch, seed=1):

		'''
		Fuction to perform label-wise mixup on a batch of data

		Parameters
		----------
		batch : tf.data.Dataset()
			The batch over which to compute the margin distance. A tuple of tf.Tensor of the form (input data, labels)
		seed : int, optional
			Random seed

		Returns
		-------
		tf.tensor
			The label-wise mixed-up batch
		'''

		np.random.seed(seed)

		def intrapolateImages(img, alpha=0.5):
			temp = np.stack([img]*img.shape[0])
			try:
				tempT =  np.transpose(temp, axes = (1,0,2,3,4))
			except:
				tempT =  np.transpose(temp, axes = (1,0,2,3))
			ret = alpha*temp + (1-alpha)*tempT
			mask = np.triu_indices(img.shape[0], 1)
			return ret[mask]

		def randomSample(batch, size):
			indices = np.random.randint(0, batch.shape[0], size=size)
			return batch[indices]

		for label in range(1+np.max(batch[1].numpy())):
			try:
				img = batch[0][batch[1]==label]
				lbl = batch[1][batch[1]==label]
				try:
					mixedBatch = np.vstack((mixedBatch, randomSample(intrapolateImages(img), img.shape[0])))
					labels = np.concatenate((labels, lbl))
				except Exception as e:
					mixedBatch = randomSample(intrapolateImages(img), img.shape[0])
					labels = lbl
			except:
				img = batch[0][batch[1]==label]
				lbl = batch[1][batch[1]==label]
				try:
					mixedBatch = np.vstack((mixedBatch, img))
					labels = np.concatenate((labels, lbl))
				except:
					mixedBatch = img
					labels = lbl

		return (tf.convert_to_tensor(mixedBatch), tf.convert_to_tensor(labels))

	# ====================================== Utility Functions ======================================

	def intermediateOutputs(self, batch, layer=None, mode=None):

		'''
		Fuction to get intermadiate feature vectors

		Parameters
		----------
		batch : tf.Tensor
			A batch of data
		layer : int, optional
			The layer for which to get intermediate features
		mode : str, optional
			'pre' to create a pre-model which takes in the input data and gives out intermediate activations,
				'post' to take in intermediate activations and give out the model predictions
	
		Returns
		-------
		tf.keras.Model()
			An extractor model
		'''

		model_ = keras.Sequential()
		model_.add(keras.Input(shape=(batch[0][0].shape)))
		for layer_ in self.model.layers:
			model_.add(layer_)

		if layer is not None and mode=='pre':
			if layer >= 0:
				extractor = keras.Model(inputs=self.model.layers[0].input,
		                        outputs=self.model.layers[layer].output)
			else:

				extractor = keras.Model(inputs=self.model.layers[0].input,
		                        outputs=self.model.layers[0].input)
		elif layer is not None and mode=='post':
			input_ = keras.Input(shape = (self.model.layers[layer].input.shape[1:]))
			next_layer = input_
			for layer in self.model.layers[layer:layer+2]:
			    next_layer = layer(next_layer)
			extractor = keras.Model(input_, next_layer)
		else:
			extractor = keras.Model(inputs=self.model.layers[0].input,
		                        outputs=[layer.output for layer in self.model.layers])

		return extractor