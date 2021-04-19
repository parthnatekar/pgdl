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
from tensorflow.keras.models import load_model
from scipy.stats import *
sys.path.append('..')
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances, calinski_harabasz_score
from sklearn.utils import check_X_y, _safe_indexing
from scipy.stats import wasserstein_distance, moment
from math import log, exp
import matplotlib.pyplot as plt
from augment import *
from computecomplexityfinal import *
import copy
import math

def complexityMargin(model, dataset, augment='standard', program_dir=None):

	'''
	Fuction to calculate margin summary measure on augmented data

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	augment : str, optional
		The type of augmentation to use ('standard', 'mixup', 'adverserial', 'adverserial+standard', 'mixup+standard')
	program_dir : str, optional
		The program directory to store and retrieve additional data

	Returns
	-------
	float
		complexity measure
	'''

	keras.backend.clear_session()
	
	np.random.seed(0)

	marginDistribution = {}
	C = CustomComplexityFinal(model, dataset, augment=augment, input_margin=False)
	for label in range(2, 3):
		marginDistribution[label], normwidth = C.computeMargins(top = label)

	score = 0

	def computeQuantiles(d):

		q1 = np.percentile(d, 25)
		q2 = np.percentile(d, 50)
		q3 = np.percentile(d, 75)

		iqr = q3 - q1

		if d[d < q1 - 1.5*iqr].size == 0:
			f_l = np.min(d)
		else:
			f_l = np.max(d[d < q1 - 1.5*iqr])

		if d[d > q3 + 1.5*iqr].size == 0:
			f_u = np.max(d)
		else:
			f_u = np.min(d[d > q3 + 1.5*iqr])

		ret = [f_l, q1, q2, q3, f_u]

		return np.array(ret)

	def computePercentiles(d):
		return np.array([np.percentile(d, p) for p in list(range(5, 95, 10))])

	def computeMoments(d):
		
		moments = [stats.moment(d, moment=ord)**(1/ord) for ord in range(1, 6)]

		moments[0] = np.mean(d)
		moments = np.nan_to_num(moments, nan=np.mean(moments))
		print(moments)
		return np.array(moments)

	for label in range(2, 3):
		for i, index in enumerate(list(marginDistribution[label].keys())):
			quantiles = np.mean(computeQuantiles(marginDistribution[label][index]))
			mean = np.nanmean(marginDistribution[label][index])
			score += mean/len(list(marginDistribution[label].keys())) 

	return -score


def complexityNorm(model, dataset, program_dir=None):

	'''
	Function to calculate norm based complexity measures

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	program_dir : str, optional
		The program directory to store and retrieve additional data

	Returns
	-------
	float
		complexity measure
	'''

	C = CustomComplexity(model, dataset, metric='batch_variance', augment='standard')
	Norm = C.getNormComplexity(norm='fro')
	score = 0

	score += Norm 
	params = np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
	print('Params:', params, model.layers[0].get_weights()[0].shape[-1])
	print('Final Score:', score, len(model.layers))
	
	return score

def complexityDB(model, dataset, program_dir=None, pool=True, use_pca = False, layer='initial', computeOver=400, batchSize=40):

	'''
	Function to calculate feature clustering based measures. Based on the sklearn implementation of DB Index.

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	program_dir : str, optional
		The program directory to store and retrieve additional data
	pool : bool, optional
		Whether to use max-pooling for dimensionality reduction, default True
	use_pca : bool, optional
		Whether to use PCA for dimensionality reduction, default False
	layer : str or int, optional
		Which layer to compute DB on. Either 'initial', for the first conv/pooling layer in the 
		model, 'pre-penultimate' for the 3rd-from-last conv/pool layer, or an int indicating the 
		layer. Defaults to 'initial'.

	Returns
	-------
	float
		complexity measure
	'''

	def check_number_of_labels(n_labels, n_samples):
	    """Check that number of labels are valid.
	    Parameters
	    ----------
	    n_labels : int
	        Number of labels
	    n_samples : int
	        Number of samples
	    """
	    if not 1 < n_labels < n_samples:
	        raise ValueError("Number of labels is %d. Valid values are 2 "
	                         "to n_samples - 1 (inclusive)" % n_labels)

	def db(X, labels):
		X, labels = check_X_y(X, labels)
		le = LabelEncoder()
		labels = le.fit_transform(labels)
		n_samples, _ = X.shape
		n_labels = len(le.classes_)
		check_number_of_labels(n_labels, n_samples)

		intra_dists = np.zeros(n_labels)
		centroids = np.zeros((n_labels, len(X[0])), dtype=float)
		for k in range(n_labels):
			cluster_k = _safe_indexing(X, labels == k)
			centroid = cluster_k.mean(axis=0)
			centroids[k] = centroid
			intra_dists[k] = np.average(pairwise_distances(
			    cluster_k, [centroid], metric='euclidean'))

		centroid_distances = pairwise_distances(centroids, metric='euclidean')

		if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
			return 0.0

		centroid_distances[centroid_distances == 0] = np.inf
		combined_intra_dists = intra_dists[:, None] + intra_dists
		scores = np.max(combined_intra_dists / centroid_distances, axis=1)
		return np.mean(scores)

	tf.keras.backend.clear_session()
	db_score = {}
	C = CustomComplexityFinal(model, dataset, augment='mixup', computeOver=computeOver, batchSize=batchSize)
	it = iter(dataset.repeat(-1).batch(C.batchSize))
	batch=next(it)
	extractor = C.intermediateOutputs(batch=batch)
	if pool == True:
		max_pool = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, padding="valid", data_format=None)
	else:
		max_pool = tf.keras.layers.Lambda(lambda x: x + 0)
	layers = []

	layer_dict = {'initial': [0, 1, 2], 'pre-penultimate': [-3, -4, -5]}

	if isinstance(layer, str):
		for l in layer_dict[layer]:
			c = list(model.get_layer(index = l).get_config().keys())
			if 'strides' in c:
				layers.append(l)
			if len(layers) == 1:
				break
	else:
		for l in [layer]:
			c = list(model.get_layer(index = l).get_config().keys())
			if 'strides' in c:
				layers.append(l)
			if len(layers) == 1:
				break

	D = DataAugmentor(batchSize = C.batchSize)
	for l in layers:
		it = iter(dataset.repeat(-1).shuffle(5000, seed=1).batch(C.batchSize))
		for i in range(C.computeOver//C.batchSize):
			tf.keras.backend.clear_session()
			batch1 = next(it)
			# batch1 = (D.augment(batch1[0]), batch1[1])
			batch2 = next(it)
			# batch2 = (D.augment(batch2[0]), batch2[1])
			batch3 = next(it)
			# batch3 = (D.augment(batch3[0]), batch3[1])
			feature = np.concatenate((max_pool(extractor(batch1[0].numpy())[l]).numpy().reshape(batch1[0].shape[0], -1), 
										max_pool(extractor(batch2[0].numpy())[l]).numpy().reshape(batch2[0].shape[0], -1), 
										max_pool(extractor(batch3[0].numpy())[l]).numpy().reshape(batch3[0].shape[0], -1)), axis = 0)
			target = np.concatenate((batch1[1], batch2[1], batch3[1]), axis = 0)
			if use_pca == True:
				pca = PCA(n_components=25)
				feature = pca.fit_transform(feature)
			try:
				db_score[l] += db(feature, target)/(C.computeOver//C.batchSize)
			except Exception as e:
				db_score[l] = db(feature, target)/(C.computeOver//C.batchSize)

	score = np.mean(list(db_score.values()))

	return(score)

def complexityMixup(model, dataset, program_dir=None,
					computeOver=500, batchSize=50):

	'''
	Function to calculate label-wise Mixup based measure

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	program_dir : str, optional
		The program directory to store and retrieve additional data
	computeOver : int
		The number of samples over which to compute the complexity measure
	batchSize: 
		The batch size

	Returns
	-------
	float
		complexity measure
	'''
	it = iter(dataset.repeat(-1).shuffle(5000, seed=1).batch(batchSize))
	batch = next(it)
	n_classes = 1+np.max(batch[1].numpy())
	batchSize = n_classes*10
	computeOver = batchSize*10
	tf.keras.backend.clear_session()
	it = iter(dataset.repeat(-1).batch(batchSize))
	N = computeOver//batchSize
	batches = [next(it) for i in range(N)]
	vr = []

	def intrapolateImages(img, alpha=0.5):
		temp = np.stack([img]*img.shape[0])
		tempT =  np.transpose(temp, axes = (1,0,2,3,4))
		ret = alpha*temp + (1-alpha)*tempT
		mask = np.triu_indices(img.shape[0], 1)
		return ret[mask]

	def choose(n, k):
	    """
	    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
	    """
	    if 0 <= k <= n:
	        ntok = 1
	        ktok = 1
	        for t in range(1, min(k, n - k) + 1):
	            ntok *= n
	            ktok *= t
	            n -= 1
	        return ntok // ktok
	    else:
	        return 

	def veracityRatio(model, batches, label, version_loss=None, label_smoothing=0.1):
		ret = []
		lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
		for b in batches:
			img = b[0][b[1]==label]
			lbl = b[1][b[1]==label]
			int_img = intrapolateImages(img)
			int_lbl = np.stack([label]*int_img.shape[0])
			int_logits = model(int_img)
			if version_loss == 'log':
				logLikelihood = lossObject(int_lbl, int_logits)
				ret.append(logLikelihood)
			elif version_loss == 'cosine':
				int_preds = tf.nn.softmax(int_logits, axis = 1)
				target = tf.one_hot(int_lbl, int_preds.shape[-1]) * (1 - label_smoothing) + label_smoothing/2
				ret.append((tf.keras.losses.CosineSimilarity()(target, int_preds)+1)/2)
			elif version_loss == 'mse':
				int_preds = tf.nn.softmax(int_logits, axis = 1)
				target = tf.one_hot(int_lbl, int_preds.shape[-1]) #* (1 - label_smoothing) + label_smoothing/2
				ret.append(tf.keras.losses.MeanSquaredError()(target, int_preds))
			else:
				int_preds = tf.argmax(int_logits, axis=1)
				ret.append(np.sum(int_preds==label)/np.size(int_preds))
		return np.mean(ret)

	for l in range(n_classes):
		try:
			vr.append(veracityRatio(model, batches, l))
		except:
			pass

	return np.mean(vr)


def complexityManifoldMixup(model, dataset, program_dir=None, 
							computeOver=1000, batchSize=50):

	'''
	Function to calculate Manifold Mixup based measure

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	program_dir : str, optional
		The program directory to store and retrieve additional data
	computeOver : int
		The number of samples over which to compute the complexity measure
	batchSize: int
		The batch size

	Returns
	-------
	float
		complexity measure
	'''

	it = iter(dataset.repeat(-1).batch(batchSize))
	N = computeOver//batchSize
	batches = [next(it) for i in range(N)]
	digress = []

	def intrapolateImages(img, alpha=0.5):
		temp = np.stack([img]*img.shape[0])
		tempT =  np.transpose(temp, axes = (1,0,2,3,4))
		ret = alpha*temp + (1-alpha)*tempT
		mask = np.triu_indices(img.shape[0], 1)
		return ret[mask]

	def multiplicativeNoise(img, std=2):
		return img*(tf.random.normal(mean=1., stddev=std, shape=img.shape, seed=1))

	def veracityRatio(model, batches, label, layer=0, version_loss=False):

		cloned_model = keras.models.clone_model(model, input_tensors=keras.Input(shape=(batches[0][0][0].shape)))
		cloned_model.set_weights(model.get_weights())
		if layer != 0:
			cloned_model.layers[layer-1].activation = keras.activations.linear
		for b in batches:
			ret = []
			img = b[0][b[1]==label]
			orig_logits = cloned_model(img)
			representation = cloned_model.layers[layer]._last_seen_input
			int_repr = intrapolateImages(representation)
			int_lbl = np.stack([label]*int_repr.shape[0])
			x = int_repr
			for i in range(layer, len(model.layers)):
				if i == layer and i != 0:
					x = keras.activations.relu(x)
					x = model.layers[i](x)
				else:
					x = model.layers[i](x)
			if version_loss:
				logLikelihood = lossObject(int_lbl, x)
				ret.append(logLikelihood/np.size(int_preds))
			else:
				int_preds = tf.argmax(x, axis=1)
				ret.append(np.sum(int_preds==label)/np.size(int_preds))
		return np.mean(ret)

	for l in range(10):
		try:
			digress.append(veracityRatio(model, batches, l, layer = 1))
		except:
			pass
	return np.mean(digress)

def complexityMixupSoft(model, dataset, program_dir=None,
					computeOver=1000, batchSize=100):

	'''
	Function to calculate Mixup based measure

	Parameters
	----------
	model : tf.keras.Model()
		The Keras model for which the complexity measure is to be computed
	dataset : tf.data.Dataset
		Dataset object from PGDL data loader
	program_dir : str, optional
		The program directory to store and retrieve additional data
	computeOver : int
		The number of samples over which to compute the complexity measure
	batchSize: int
		The batch size

	Returns
	-------
	float
		complexity measure
	'''

	it = iter(dataset.repeat(-1).shuffle(5000, seed=1).batch(batchSize))
	batch = next(it)
	n_classes = 1+np.max(batch[1].numpy())
	batchSize = n_classes*10
	computeOver = batchSize*10
	tf.keras.backend.clear_session()
	it = iter(dataset.repeat(-1).batch(batchSize))
	N = computeOver//batchSize
	batches = [next(it) for i in range(N)]
	np.random.seed(0)
	vr = []

	def intrapolateImages(img, lbl, alpha=0.5):
		indices = np.random.randint(0, img.shape[0], size = img.shape[0])
		img = img*alpha + img[indices]*(1-alpha)
		lbl = lbl*alpha + lbl[indices]*(1-alpha)
		return (tf.convert_to_tensor(img), tf.convert_to_tensor(lbl))

	def choose(n, k):
	    """
	    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
	    """
	    if 0 <= k <= n:
	        ntok = 1
	        ktok = 1
	        for t in range(1, min(k, n - k) + 1):
	            ntok *= n
	            ktok *= t
	            n -= 1
	        return ntok // ktok
	    else:
	        return 

	def veracityRatio(model, batches, version_loss=None, label_smoothing=0.1):
		ret = []
		lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
		for b in batches:
			int_img, int_lbl = intrapolateImages(b[0].numpy(), b[1].numpy())
			int_logits = model(int_img)
			if version_loss == 'log':
				logLikelihood = lossObject(int_lbl, int_logits)
				ret.append(logLikelihood)
			elif version_loss == 'cosine':
				int_preds = tf.nn.softmax(int_logits, axis = 1)
				target = tf.one_hot(int_lbl, int_preds.shape[-1]) * (1 - label_smoothing) + label_smoothing/2
				ret.append((tf.keras.losses.CosineSimilarity()(target, int_preds)+1)/2)
			elif version_loss == 'mse':
				int_preds = tf.nn.softmax(int_logits, axis = 1)
				target = tf.one_hot(int_lbl, int_preds.shape[-1]) #* (1 - label_smoothing) + label_smoothing/2
				ret.append(tf.keras.losses.MeanSquaredError()(target, int_preds))
			else:
				int_preds = tf.argmax(int_logits, axis=1)
				ret.append(np.sum(int_preds==label)/np.size(int_preds))
		return np.mean(ret)

	vr.append(veracityRatio(model, batches, version_loss='log'))

	return np.mean(vr)