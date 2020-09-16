#!/usr/bin/env python
# coding: utf-8

from IPython.display import clear_output

import numpy as np
import pandas as pd
import os
import glob
import logging
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from importlib import reload
from tqdm import tqdm
import pickle
from pygmo.core import hypervolume
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from dataloader.fairness_datahandler import FairnessDataHandler
from dataloader.fairness_dataset import CustomDataset
from models.linear_regression import LinearRegression
from models.nn1 import NN1
from models.nn2 import NN2
from loss.losses import *
from metric.metrics import *
from trainer import Trainer
from single_objective_trainer import SingleObjectiveTrainer
from validator import Validator
from loss.loss_class import Loss

from public_experiments.pareto_utils import *


device = torch.device('cpu')
get_ipython().run_line_magic('matplotlib', 'inline')


X = np.load('data/compas/X.npy')
y = np.load('data/compas/y.npy')
X1 = torch.from_numpy(X).float().to(device)
y1 = torch.from_numpy(y).float().to(device)

input_dimension = X1.shape[1]

data = CustomDataset(X1, y1)

total_samples = X1.shape[0]
train_samples = 3000
val_samples = 2000
test_samples = int(total_samples - train_samples - val_samples)

train_data, val_data, test_data = random_split(data, [train_samples, val_samples, test_samples])



# set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('compas')
logger.setLevel(logging.INFO)

# Multi-objective setup
zero_weight = (y[:,0] == 0).sum()
one_weight = (y[:,0] == 1).sum()
n = max(zero_weight, one_weight)

weights = torch.tensor([np.float(n)/zero_weight, np.float(n)/one_weight], dtype=torch.float, device=device)

save_to_path = 'saved_models/compas/'

input_dim = train_data.dataset.x.shape[1]
lr = 5e-3

def build_model_compas(fairness_notion='ddp'):
    model = NN1(input_dimension=input_dim)
    model.to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    performance_loss = BCELoss(name='compas_bce')
    race_loss_EOP = TPRLoss(name='compas_DPP_race', reg_lambda=0.1, reg_type='tanh')
    race_loss_DDP = DPLoss(name='compas_DPP_race', reg_lambda=0.1, reg_type='tanh')
    if(fairness_notion == 'ddp'):
        losses = [performance_loss, race_loss_DDP]
    elif(fairness_notion == 'deo'):
        losses = [performance_loss, race_loss_EOP]
    elif(fairness_notion == 'both'):
        losses = [performance_loss, race_loss_DDP, race_loss_EOP]
    
    return model, optimizer, losses

def compas_multi_fairness(fairness_notion='ddp'):
    # metrics
    accuracy = Accuracy(name='accuracy')
    ddp = DPDiff(name='DDP')
    deo = TPRDiff(name='DEO')
    if(fairness_notion == 'ddp'):
        validation_metrics = [accuracy, ddp]
    elif(fairness_notion == 'deo'):
        validation_metrics = [accuracy, deo]
    elif(fairness_notion == 'both'):
        validation_metrics = [accuracy, ddp, deo]

    scores_mo_zenith_compas = []
    losses_mo_zenith_compas = []

    matches_zenith = []
    matches_fair = []

    for i in tqdm(range(10)):

        train_data, val_data, test_data = random_split(data, [train_samples, val_samples, test_samples])
        data_handler = FairnessDataHandler('compas', train_data, val_data, test_data)

        model, optimizer, losses = build_model_compas(fairness_notion)

        files = glob.glob(save_to_path + '*')
        for f in files:
            os.remove(f)

        trainer_compas = Trainer(data_handler, model, losses, validation_metrics, save_to_path,                      params='yaml_files/trainer_params_compas.yaml', optimizer=optimizer)

        trainer_compas.train()
        scores_val = to_np(trainer_compas.pareto_manager._pareto_front)
        chosen_score_zenith, idx_zenith = get_solution(scores_val)

        ####### closest to zenith point #############
        model_val = NN1(input_dimension=input_dim)
        model_val.to(device)

        match_zenith = '_'.join(['%.4f']*len(chosen_score_zenith)) % tuple(chosen_score_zenith)
        files = glob.glob(save_to_path + '*')
        for f in files:
            if(match_zenith in f):
                model_val.load_state_dict(torch.load(f))
                continue

        test_len = data_handler.get_testdata_len()
        test_loader = data_handler.get_test_dataloader(drop_last=False,                                                       batch_size=test_len)
        test_validator = Validator(model_val, test_loader, validation_metrics, losses)

        test_metrics, test_losses = test_validator.evaluate()

        scores_mo_zenith_compas.append(test_metrics)
        losses_mo_zenith_compas.append(test_losses)
        
        scores_mo_compas = np.array(scores_mo_zenith_compas)

        clear_output(wait=True)
        
    if(fairness_notion == 'both'):
        print('The error has mean ', 1-np.mean(scores_mo_compas[:,[0]]),              ' and standard deviation ', np.std(scores_mo_compas[:,[0]]))
        print('The ddp has mean ', 1-np.mean(scores_mo_compas[:,[1]]),              ' and standard deviation ', np.std(scores_mo_compas[:,[1]]))
        print('The deo has mean ', 1-np.mean(scores_mo_compas[:,[2]]),              ' and standard deviation ', np.std(scores_mo_compas[:,[2]]))
    else:        
        print('The error has mean ', 1-np.mean(scores_mo_compas[:,[0]]),              ' and standard deviation ', np.std(scores_mo_compas[:,[0]]))
        print('The fairness has mean ', 1-np.mean(scores_mo_compas[:,[1]]),              ' and standard deviation ', np.std(scores_mo_compas[:,[1]]))






