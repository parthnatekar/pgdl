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

class CustomComplexityFinal:

	def __init__(self, model, ds, rootpath=None, mid=None, computeOver = 500, batchSize = 50, basename=None, metric='batch_variance', augment='standard', penalize=True, input_margin=False, network_scale = False):
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
 
	# ====================================== Functions for Margin Based Solution =====================================

	def computeMargins(self, top = 2):

		it = iter(self.dataset.shuffle(5000, seed=1).batch(self.batchSize))

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
			elif self.augment == 'adv_aug':
				D = DataAugmentor(batch[0], batchSize = self.batchSize)
				batch_ = (D.augment(), batch[1])
				batch_ = (self.getAdverserialBatch(batch_), batch_[1])
			elif self.augment == 'mixup':
				batch_ = self.batchMixupLabelwise(batch)
			elif self.augment == 'mixup+standard':
				D = DataAugmentor(batch[0], batchSize = self.batchSize)
				batch_ = (D.augment(), batch[1])
				batch_ = self.batchMixupLabelwise(batch_)
				
			for layer in self.layers:
				if self.augment is not None:
					grads, inter = self.distancefromMargin(batch_, layer+1, top)
				else:
					grads, inter = self.distancefromMargin(batch, layer+1, top)

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
			totalVariance[layer] = (np.mean(np.var(totalVarianceTensor[layer].reshape(totalVarianceTensor[layer].shape[0], -1), axis = 0)))**(1/2)
			normWidth[layer] = math.sqrt(np.prod(totalVarianceTensor[layer].shape[1:]))
			if self.metric == 'batch_variance':
				marginDistribution[layer] = np.array(marginDistribution[layer])/(np.array(totalVariance[layer])+1e-7) #/np.sqrt(m_factor) #/totalVarianceTensor[layer].shape[1:][0]
			elif self.metric == 'original':
				marginDistribution[layer] = np.array(marginDistribution[layer])/np.array(totalVariance[layer]) #/np.sqrt(m_factor) #/totalVarianceTensor[layer].shape[1:][0]
			
		return marginDistribution, normWidth

	def distancefromMargin(self, batch, layer, top = 2):

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

	def batchMixupLabelwise(self, batch, seed=1):

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