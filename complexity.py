
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
from computecomplexityfinal import *
from complexitymeasures import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import * 
from augment import *

def complexity(model, dataset, program_dir):
	'''
	Wrapper Complexity Function to combine various complexity measures

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
	
	marginScore = complexityMargin(model, dataset, augment = 'standard', program_dir=program_dir)
	# DBScore = complexityDB(model, dataset, program_dir=program_dir)
	# tf.keras.backend.clear_session()
	# mixupScore = complexityMixupSoft(model, dataset, program_dir=program_dir)	
	print('-------Final Scores---------', marginScore)
	return marginScore