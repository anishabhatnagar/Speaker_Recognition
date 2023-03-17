import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageFile
from torchvision import models
import torch
# Neural networks can be constructed using the torch.nn package.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import time
import copy

import torchaudio
import re




''' 
GENERATE DIFFERENT FEATURES FROM KAGGLE SPEAKER RECOGNITION DATASET
'''

def generatefeatures_from_kaggle_data(root_dir, feature):
	tensor_list=[]
	label_list=[]
	fixed_sample_rate=16000

	path=root_dir
	label_folders=os.listdir(path)
	for label in label_folders:
		path=root_dir+"/"+label
		# level5 - Access files
		files=os.listdir(path)
		for f in files:
			label_list.append(label)

			path=root_dir+"/"+label+"/"+f
			wav, sr = torchaudio.load(path)
			#wav = wav[startTime:endTime]
			##Code for mel spec ----
			if feature == "MELSPEC" :
				feat=melspec(wav, fixed_sample_rate)
				tensor_list.append(feat)
				
			elif feature == "Classic Spectrogram":
				feat=clspec(wav)
				tensor_list.append(feat)
			elif feature == "MFCC":
				feat=mfcc(wav, fixed_sample_rate)
				tensor_list.append(feat)
			
	Featdict={"Feature":tensor_list,"label":label_list}
	return Featdict

'''
MEL - SPEC
'''

def melspec(wav,fixed_sample_rate):
	melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=fixed_sample_rate, n_mels=128)
	melspectrogram = melspectrogram_transform(wav)
	##Amplitude to DB transform
	melspectrogram_db_transform = torchaudio.transforms.AmplitudeToDB()
	melspectrogram_db=melspectrogram_db_transform(melspectrogram)
	melspectrogram_db_np=melspectrogram_db.numpy()
	return melspectrogram_db_np


'''
Classic spectrogram
'''

def clspec(wav):
	spectrogram_transform = torchaudio.transforms.Spectrogram()
	spectrogram=spectrogram_transform(wav)
	spectrogram_db_transform = torchaudio.transforms.AmplitudeToDB()
	spectrogram_db = spectrogram_db_transform(spectrogram)
	spectrogram_db_np = spectrogram_db.numpy()
	return spectrogram_db_np

'''
MFCC
'''
def mfcc(wav,fixed_sample_rate):
	mfcc_module = torchaudio.transforms.MFCC(sample_rate=fixed_sample_rate, n_mfcc=20, melkwargs={"n_fft": 2048, "hop_length": 512, "power": 2})
	torch_mfcc = mfcc_module(wav)
	mfcc_feat_np = torch_mfcc.numpy()
	return mfcc_feat_np


''' 
CREATING MEL SPEC FROM TIMIT
'''
def generateMELSPEC_from_TIMIT(root_dir):
	tensor_list=[]
	label_list=[]
	fixed_sample_rate=16000

	# startTime = 0
	# endTime = 167

	tt=os.listdir(root_dir)
	#level 1
	for t in tt:
	    #level 2 - train/test
	    path=root_dir+"/"+t
	    
	    #level3 - dialect folders
	    dialect_folders=os.listdir(path)
	    for dialect in dialect_folders:
	        path=root_dir+"/"+t+"/"+dialect
	        
	        #level4 - label folders
	        label_folders=os.listdir(path)
	        for label in label_folders:
	            path=root_dir+"/"+t+"/"+dialect+"/"+label
	            
	            # level5 - Access files
	            files=os.listdir(path)
	            for f in files:
	                if re.findall("WAV$", f):
	                    path=root_dir+"/"+t+"/"+dialect+"/"+label+"/"+f
	                else:
	                    continue
	                
	                label_list.append(label)
	                
	                ## Read wav file
	                wav, sr = torchaudio.load(path)
	                #wav = wav[startTime:endTime]

	                ##Code for mel spec ----
	                melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=fixed_sample_rate, n_mels=128)
	                melspectrogram = melspectrogram_transform(wav)
	                
	                ##Amplitude to DB transform
	                melspectrogram_db_transform = torchaudio.transforms.AmplitudeToDB()
	                melspectrogram_db=melspectrogram_db_transform(melspectrogram)
	                melspectrogram_db_np=melspectrogram_db.numpy()
	                tensor_list.append(melspectrogram_db_np)
	
	# tensor_list=np.array(tensor_list)
	# label_list=np.array(label_list)
	# print("tensor_list type: ",type(tensor_list))
	MELdict={"Mel_DB_spec":tensor_list,"label":label_list}
	return MELdict


