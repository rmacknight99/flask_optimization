#!/usr/bin/env python

import sys
import os
import shutil
import math
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import pandas as pd
from olympus import Olympus
from olympus import list_planners
from olympus import Plotter
from olympus import Dataset
from olympus import ParameterSpace, Parameter
from olympus import Emulator, Campaign, Planner
from olympus.models import BayesNeuralNet
import json
import argparse

def truncate(n, decimals=0):
    mult = 10 ** decimals
    return int(n * mult) / mult

    # define function to create a dataset in olympus using
    # either a config.json file or csv file with column headers 
    
def make_dataset(name):
    """
    Create Olympus dataset object and write config, description, and data files

    Args:
            name: name of dataset (hplc)

    Returns:
            dataset: Olympus dataset object
    """
    try:
        
        f = open('datasets/dataset_{}/config.json'.format(name))
        config = json.load(f)
        parameters_dict = config['parameters']
        parameters_list = []

        for i in parameters_dict:
            parameters_list.append(i['name'])
        target_dict = config['measurements']
        target_list = []

        for i in target_dict:
            target_list.append(i['name'])
            
        columns = parameters_list+target_list
        target_ids = target_list
        mydata = pd.read_csv('datasets/dataset_{}/data.csv'.format(name))

        dataset = Dataset(data=mydata,target_ids = target_ids)
            
    except:
        
        print('------no config file present------\n')            
        print('------for correct behavior ensure the csv file has column headers------\n')

        try:
            os.mkdir('datasets/dataset_{}/'.format(name))
        except:
            pass

        mydata = pd.read_csv('datasets/dataset_{}/data.csv'.format(name))
        target_ids = mydata.columns.values.tolist()[-1]

        if isinstance(target_ids[0],str):
            print('------headers provided------\n')

        dataset = Dataset(data=mydata,target_ids=[target_ids])

    param_space=ParameterSpace()
    for p in dataset.features:

        low = np.min(dataset.data[p])
        high = np.max(dataset.data[p])
        low = truncate(low, 3)
        high = truncate(high, 3)

        print("\n")
        print("parameter: {}, domain: [{},{}]".format(p,low,high))
        
        param = Parameter(kind = 'continuous',name = p,low = low,high = high)
        param_space.add(param)
        
    dataset.set_param_space(param_space)
    print('------copying dataset to disk------\n')

        # remove original dataset directory containing data.csv file with headers
        # and rewrite the directory with the data (no headers), config, and description files

    shutil.rmtree('datasets/dataset_{}/'.format(name))
    dataset.to_disk('datasets/dataset_{}/'.format(name))

    return dataset

def train_emulator(name,batch_size,reg,hidden_act,out_act,max_epochs,feature_transform,target_transform,save=True): # train bayesian neural network emulator
    """
    Train a BNN experiment simulator 

    Args:
            name: name of dataset
            batch_size: number of training points in training batch
            reg: ??? a parameter involved in the loss function
            hidden_act: hidden activation function
            out_act: output activation function
            max_epochs: number of training epochs
            feature_transform: normalize
            target_transform: normalize
            save: boolean, whether or not you would like to save your emulator

    Returns:
            bnn: BNN model architecture
            emulator: trained BNN experiment simulator
            dataset: Olympus dataset object
            scores: cross validation scores for given model architecture
    """
    bnn = BayesNeuralNet(hidden_depth = 2,hidden_nodes = 12,hidden_act = hidden_act,out_act = out_act,
                         batch_size = batch_size,reg = reg,max_epochs = max_epochs)
    
    dataset = make_dataset(name)
    
    emulator = Emulator(dataset = dataset,model = bnn,feature_transform = feature_transform,target_transform = target_transform)

    if save:# save the emulator in the emulators directory
        scores = emulator.cross_validate()
        emulator.train()
        emulator.save('emulators/emulator_{}_BayesNeuralNet'.format(name))

    else:# retrieve scores of cross validation
        scores = emulator.cross_validate()

    # this block of code checks to see if a bank directory exists, if not, a bank directory is created
    # based on the hyperparameters of the bnn, the scores are then written to file 
        
    if os.path.isdir('bank/{}_{}_{}_{}_{}'.format(name,hidden_act,out_act,feature_transform,target_transform)):

        print('------bank directory exists------\n')

        if os.path.isfile('bank/{}_{}_{}_{}_{}/scores_{}_{}_{}.txt'.format(name,hidden_act,out_act,feature_transform,target_transform,batch_size,reg,max_epochs)):
            os.remove('bank/{}_{}_{}_{}_{}/scores_{}_{}_{}.txt'.format(name,hidden_act,out_act,feature_transform,target_transform,batch_size,reg,max_epochs))
            print('------previous scores being overwritten------\n')
        
    else:

        os.mkdir('bank/{}_{}_{}_{}_{}'.format(name,hidden_act,out_act,feature_transform,target_transform))
        print('------creating new bank directory------')
        
    for key in scores.keys():

        values = scores.get(key)

        with open('bank/{}_{}_{}_{}_{}/scores_{}_{}_{}.txt'.format(name,hidden_act,out_act,feature_transform,target_transform,batch_size,reg,max_epochs),'a') as f:
            f.write('{}:{}\n'.format(key,values))

    return bnn,emulator,dataset,scores
