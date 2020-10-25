
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
	marginScore = complexity_but_simple(model, dataset, augment = 'standard', program_dir=program_dir)
	# DBScore = complexityDB(model, dataset, program_dir=program_dir)
	# tf.keras.backend.clear_session()
	mixupScore = complexityMixup(model, dataset, program_dir=program_dir)	
	print('-------Final Scores---------', marginScore, mixupScore)
	return marginScore*mixupScore