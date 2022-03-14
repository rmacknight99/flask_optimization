# import python packages
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
from olympus.emulators.emulator import load_emulator
from olympus.models import BayesNeuralNet
from olympus.objects.object_parameter_vector import ObjectParameterVector
import json
import argparse

def get_headers(config): # get header names by using config file in dataset directory
    """
    Retrieve the "headers" = parameters + targets for a datsets using the config file.

    Args:
            config: path to config file for dataset.

    Returns:
            headers: parameters + targets for dataset
    
    """
    f = open(config)
    config = json.load(f)
    parameters_dict = config['parameters']
    parameters_list = []

    for i in parameters_dict: # get parameter names
        parameters_list.append(i['name'])

    target_dict = config['measurements']
    target_list = []

    for i in target_dict: # get target names
        target_list.append(i['name'])

    headers = parameters_list+target_list # get header names

    return headers # return header names
    
def load_observations(path_to_data,names): # load observations by using data file in dataset directory
    """
    Retreive observations from csv file with headers

    Args:
            path_to_data: path to csv file containing parameter and target measurements
            names: list of header names for the data file (parameter + target names)

    Returns:
            observations: parameter measurements
            values: target measurements
            
    """
    headers = names
    df = pd.read_csv(path_to_data,names = headers)
    obj = headers.pop()
    observations = []
    values = []
    rows = len(df.index)
        
    for i in range(rows):

        if i != 0:
            row = df.iloc[i]
            sample = []

            for p in headers:
                sample.append(row[p])

            value = [row[obj]]
            values.append(value)

            observations.append(sample)

        else:
            continue
        
    return observations,values # return observation parameters and objective function value

def append_data(old_data,new_data,names): # append suggested observations to old observations (this is optional)
    """
    Write updated csv data file with new suggested observations

    Args:
            old_data: input data to the optimization campaign
            new_data: output data of the optimization campaign
            names: parameters + targets for dataset

    Returns:
            None
    """
    old_data = pd.read_csv(old_data,names = names)
    new_data = pd.read_csv(new_data)
    all_data = pd.concat([old_data,new_data],ignore_index = True)

    all_data.to_csv('updated_data.csv',index = False)

def archive_data(path,campaign_number): # archive the data files used to load observations for current optimization campaign
    """
    Store previous dataset files and update data file to include suggested observations

    Args:
            path: the path to the dataset directory "datasets/dataset_{name}/"
            campaign_number: the optimization campaign number

    Returns:
            None
    """
    archive_number = campaign_number-1
    os.mkdir('{}/archive_{}'.format(path,archive_number))
    shutil.move('{}/description.txt'.format(path),'{}/archive_{}/description.txt'.format(path,archive_number))
    shutil.move('{}/config.json'.format(path),'{}/archive_{}/config.json'.format(path,archive_number))
    shutil.move('{}/data.csv'.format(path),'{}/archive_{}/data.csv'.format(path,archive_number))
    shutil.move('updated_data.csv','{}/data.csv'.format(path))

def optimize(dataset,name,algorithm,goal,max_iter,campaign_number,emulator=False): # optimize using previous observations from dataset, algorithm, and goal
    """
    Run optimization campaign on dataset using a given algorithm and goal

    Args:
            dataset: Olympus dataset object
            name: name of the dataset you would like to run your campaign on
            algorithm: name of optimization algorithm (Phoenics)
            goal: maximize or minimize
            max_iter: number of suggested observations
            campaign_number: for consecutive optimization campaigns
            emulator: boolean, whether or not you would like to load an emulator and label
                      suggested observations

    Returns:
            df: pandas dataframe containing the suggested observations
    """
    if emulator:
        model = load_emulator('emulators/emulator_{}_BayesNeuralNet'.format(name)) # load emulator from emulator directory
    
    f = open('datasets/dataset_{}/config.json'.format(name)) # load config file from dataset directory
    config = json.load(f)

    vs = [Parameter().from_dict(feature) for feature in config["measurements"]] # create value space
    value_space = ParameterSpace()
    value_space.add(vs)

    names = get_headers('datasets/dataset_{}/config.json'.format(name)) # get names of features and targets 
    obs,vals = load_observations('datasets/dataset_{}/data.csv'.format(name),names) # load observations and values

    observations = []
    values = []

    for i,e in enumerate(obs): # create observations and values lists from loaded observations and values

        if i % 1 == 0:
            observations.append(e)
            values.append(vals[i])

    if len(values)==len(observations): # check to see if the number of values matches the number of observations

        print('samples: {}\n'.format(len(values)))

    planner = Planner(algorithm,goal=goal) # initialize olympus planner 
    campaign = Campaign() # initialize olympus campaign
    opt_campaign = Campaign() # initialize second olympus campaign just to track the suggested data
    param_space = dataset.param_space # set dataset parameter space

    for i,e in enumerate(observations): # transform parameters and values to ObjectParameterVector

        params = ObjectParameterVector().from_array(e,param_space)
        value = ObjectParameterVector().from_array(values[i],value_space)
        campaign.add_observation(params, value) # add observations to campaign

        planner.set_param_space(dataset.param_space) # set parameter space for the optimization planner

    for i in range(max_iter): # run optimization campaign

        print(f"Iter {i+1}\n------")
        params = planner.recommend(observations = campaign.observations) # recommend new observations while considering previous observations
        print('Parameters:', params)

        if emulator:
            values = model.run(params.to_array(), return_paramvector=True) # run emulator on recommended observations to get value
        else:
            values = [0]

        print('Values:', values[0])
        campaign.add_observation(params, values) # add new observation to campaign
        opt_campaign.add_observation(params, values) # add new observation to optimization campaign
        print()

    print('no emulator given -> no values predicted\n')
    opt_vals = opt_campaign.values
    opt_params = opt_campaign.params
    flat_vals = [i for sublist in opt_vals for i in sublist]

    data = []
    for i,e in enumerate(opt_params): # create list of data points suggested by optimization algorithm
        data_point = np.concatenate((e,opt_vals[i])).tolist()
        data.append(np.round_(data_point,4))

    names = get_headers('datasets/dataset_{}/config.json'.format(name))
    df = pd.DataFrame(data=data,columns=names)

    try: # remove campaign directory 

        shutil.rmtree('campaigns/campaign_{}/'.format(name))

    except:
        pass

    os.mkdir('campaigns/campaign_{}'.format(name)) # create campaign directory
    df.to_csv('campaigns/campaign_{}/opts.csv'.format(name),index=False) # create csv file of recommended observations and values
    append_data('datasets/dataset_{}/data.csv'.format(name),'campaigns/campaign_{}/opts.csv'.format(name),names) # append recommended observations
    archive_data('datasets/dataset_{}'.format(name),campaign_number) # archive data 
    
    return df
