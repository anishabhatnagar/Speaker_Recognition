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

from Features import generatefeatures_from_kaggle_data

import time
import copy

import torchaudio
import re

def get_max_len(data):
	length=[]

	for each in data:
		s = each.size()
		length.append(s[2])

	return(max(length))

def padding(tensor,max_len):
	i, j, k = tensor.size(0), tensor.size(1),tensor.size(2)
	features = torch.cat((tensor, torch.zeros((i, j, max_len - k))),2)

	return features

def my_collate(batch):
    data = [item['t_name'] for item in batch]
    # print(type(data))
    # length = len(data)
    # print(length)
    max_length = get_max_len(data)
    
    # data[0]=torch.transpose(data[0],1,2)
    # data[0]=torch.transpose(data[0],0,2)
    # data[0]=torch.transpose(data[0],1,2)
    # print(data[0].shape)
    x=data[0]
    x=padding(x,max_length)
    for ii in range(1,128):
    	# print(ii)
    	# data[ii]=torch.transpose(data[ii],1,2)
    	# data[ii]=torch.transpose(data[ii],0,2)
    	# data[0]=torch.transpose(data[0],1,2)
    	data[ii]=padding(data[ii],max_length)
    	# print(type(data[ii]))
    	# print(data[ii].shape)
    	x=torch.cat((x,data[ii]),0)
    # data=torch.tensor(data)
    # print("data: ", data)
    x=x.unsqueeze(1)
    # x.unsqueeze(1)
    # print("x.shape: ",x.shape)
    target = [item['label'] for item in batch]
    # print("target: ",target)
    target = torch.LongTensor(target)
    # print("target.shape: ",target.shape)
    return x, target



def converter(instr):
	return np.fromstring(instr,sep=' ')


'''

CREATING CUSTOM DATASET CLASS

'''

class Spectogram_Dataset(Dataset):
	
	def __init__(self, file_paths, labels, transform=None):
		#initialization
		self.file_paths = file_paths
		self.labels = labels
		self.transform = transform
	
	def __len__(self):
		#gives length of set
		return len(self.file_paths)
	
	def __getitem__(self, index):
		#t_name =self.file_paths#.loc[index]
		# t_name=np.asarray(t_name,dtype='float64')
		t_name = torch.from_numpy(self.file_paths[index])
		# t_name = torch.tensor(t_name)
		# t_name.reshape(128,167)
		# torch.reshape(ten, (2, 3))
		# t_name = t_name.unsqueeze(0)  # if torch tensor
		# t_name = t_name.unsqueeze(1)  # if torch tensor
		# print("type t_name: ",type(t_name))
		label = torch.tensor(self.labels[index])
		if self.transform is not None:
			t_name = self.transform(t_name)
		sample={'t_name':t_name,'label': label}
		return sample



''' 
sPLITTING USING INDICES HELPER FUNCTION
'''
def split_using_indices(dataset_size):

	print("\ndataset_size = ",dataset_size)
	print("\nSPLITTING SET")
	indices = list(range(dataset_size))

	split_1 = int(np.floor(split[0] * dataset_size))
	#print(split_1)
	if shuffle_dataset :
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)
	train_indices, remaining_indices = indices[:split_1], indices[split_1:]

	remaining_dataset_size= int(dataset_size-len(train_indices))
	#print(remaining_dataset_size)
	split_2= int(np.floor(split[1] * remaining_dataset_size))
	#print(split_2)
	test_indices, val_indices = remaining_indices[split_2:], remaining_indices[:split_2]

	# train_sampler = SubsetRandomSampler(train_indices)
	# valid_sampler = SubsetRandomSampler(val_indices)
	# test_sampler = SubsetRandomSampler(test_indices)

	print("training set size: ")
	print(len(train_indices))
	print("validation set size: ")
	print(len(val_indices))
	print("testing set size: ")
	print(len(test_indices))

	return(train_indices,test_indices,val_indices)



''' 
SPLITTING USING INDICES HELPER FUNCTION
'''
def split_with_sklearn(X,Y, random_seed):
	train_data, Rem_data, train_labels, Rem_labels = train_test_split(X, Y, test_size=0.4, 
																train_size=0.6, random_state=random_seed, shuffle=True)
	Val_data, Test_data, Val_labels, Test_labels = train_test_split(Rem_data, Rem_labels, test_size=0.3, 
																train_size=0.7, random_state=random_seed, shuffle=True)

	return(train_data, train_labels, Val_data, Val_labels, Test_data, Test_labels)


''' 
CREATING DATALOADERS 
'''



def createloader(Dir,batch_size,shuffle_dataset,random_seed,split, feat_name):
	# to load any image having errors
	# ImageFile.LOAD_TRUNCATED_IMAGES = True

	# print("\n")
	# #Reading the csv file as dataframe
	# print("reading csv  file")
	# data=pd.read_csv("TensorDF.csv")#, converters = {'Mel_DB_spec' : converter} )


	# root_dir = "/home/ubuntu/TIMIT"
	# root_dir = "/home/ubuntu/16000_pcm_speeches"
	root_dir=Dir
	print(root_dir)

	'''

	CREATING mel_db TENSORS AT RUNTIME

	'''
	print("\ngenerate ",feat_name," at runtime\n\nTHIS MAY TAKE A FEW MINUTES\n\n")
	# data=generateMELSPEC_from_TIMIT(root_dir)

	data = generatefeatures_from_kaggle_data(root_dir, feat_name)

	#Encoding the labels
	print("encoding labels")
	lb = LabelEncoder()
	encoded_labels = lb.fit_transform(data['label'])
	# onehot_encoder = OneHotEncoder(drop='first', sparse=False)
	# print("integer_encoded", integer_encoded)
	# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	# print("integer_encoded", integer_encoded)
	# for i in integer_encoded:
	# 	print(i)
	
	# encoded_labels = onehot_encoder.fit_transform(integer_encoded)
	# print("encoded_labels", encoded_labels)
	# for i in encoded_labels:
		# print(i)
		# print(i.size)



	print("\n")
	print("DATA")
	#print (data.head())
	# print (data)

	# print(type(data.loc[2]['Mel_DB_spec']))
	# print(data.loc[2]['Mel_DB_spec'])

	print("\n")
	#Number of speakers
	speakers=np.unique(encoded_labels)
	speaker_count=len(speakers)
	print("number of speaker: ",speaker_count)


	#Number of Voice samples
	samples= len(data['Feature'])
	print ("number of samples in the dataset: ", samples)


	''' 

	Splitting the dataset

	'''
	#using uindidces
	# dataset_size = len(data['label'])
	# train_indices,test_indices,val_indices=split_using_indices(dataset_size)

	#using sklearn
	Train_data ,Train_labels, Val_data, Val_labels, Test_data, Test_labels = split_with_sklearn(data["Feature"],
																								encoded_labels , random_seed) 


	''' 

	calling Spectrogram dataset class 

	'''

	print("\n")
	BASE_PATH= "/home/ubuntu/GRAPHS/melDB"
	print("BASE_PATH",BASE_PATH)
	# print("applying transforms on dataset")
	# dataset = Spectogram_Dataset(data["Mel_DB_spec"],data["encoded_labels"])#,transform)
	# print("type of dataset[55]: ", type(dataset[55]))
	# print("type of dataset: ", type(dataset))
	# print(dataset[55])
	# print(dataset[22])
	# print(dataset[10])

	Traindataset = Spectogram_Dataset(Train_data, Train_labels)
	print("train_Dataset created")
	Valdataset = Spectogram_Dataset(Val_data, Val_labels)
	print("val dataset created")
	Testdataset = Spectogram_Dataset(Test_data, Test_labels)

	print("\n")


	''' 

	Creating Dataloaders

	'''

	print("\ncreating DataLoaders")
	
	train_dataloader = torch.utils.data.DataLoader(Traindataset, batch_size=batch_size,
	                                           collate_fn=my_collate, drop_last=True) #sampler=train_sampler
	print("train dataloader created")
	validation_dataloader = torch.utils.data.DataLoader(Valdataset, batch_size=batch_size,
	                                                collate_fn=my_collate, drop_last=True) #sampler=valid_sampler, 
	print("Validation dataloader created")
	test_dataloader = torch.utils.data.DataLoader(Testdataset, batch_size=batch_size,
	                                                    collate_fn=my_collate, drop_last=True) #sampler=test_sampler, 
	print("test dataloader created")

	#dataloaders={"train":train_loader,"val":validation_loader}

	print("DataLoaders created")
	return (train_dataloader,validation_dataloader,test_dataloader,speaker_count)



'''

main

'''


if __name__ == '__main__':
	##change feat name to create different features :
	#"MELSPEC" "Classic Spectrogram" "MFCC":
	feat_name="MFCC"
	root="/home/ubuntu/16000_pcm_speeches"
	t,v,ts,sp = createloader(Dir=root ,batch_size=128,shuffle_dataset =True,random_seed=42,split=[0.6,0.5], feat_name=feat_name)
	print(type(t))
	print(type(v))
	print(type(ts))
	print(type(sp))
	print("speakercount: ",sp)
